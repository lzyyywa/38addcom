import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math  # 用于计算对数权重

# 引入安全且对齐纯空间特征的底层算子
from utils.lorentz import pairwise_dist, half_aperture, oxy_angle

# ==============================================================================
# 核心损失函数实现 (严格对齐 H^2EM)
# ==============================================================================

class HierarchicalEntailmentLoss(nn.Module):
    """
    层次蕴含损失 (Entailment Cone Loss)
    严格对齐 H^2EM 公式 (11) 与 (12)
    """
    def __init__(self, K=0.1):
        super().__init__()
        self.K = K

    def forward(self, child, parent, c):
        # 1. 提取外角 theta (Eq. 12)
        theta = oxy_angle(parent, child, curv=c).unsqueeze(1)               
        
        # 2. 提取包含锥半角 omega(p)
        alpha_parent = half_aperture(parent, curv=c, min_radius=self.K).unsqueeze(1) 
        
        # 3. 严格对齐 Eq. 11: L_ent(p,q) = max(0, angle(p,q) - omega(p))
        # 注: 已遵循 H^2EM 原始公式，去除了其他文献中常见的深度惩罚项
        loss_cone = F.relu(theta - alpha_parent)

        return loss_cone.mean()


class DiscriminativeAlignmentLoss(nn.Module):
    """
    判别对齐损失 (Discriminative Alignment Loss)
    严格对齐 H^2EM 公式 (14) 及其附录的难负样本加权机制
    """
    def __init__(self, temperature=0.07, hard_weight=3.0):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_hyp, t_hyp, c, batch_verb, batch_obj):
        # 1. 使用内置 pairwise_dist 计算距离
        dist = pairwise_dist(v_hyp, t_hyp, curv=c)
        logits = -dist / self.temperature
        
        B = v_hyp.size(0)
        
        # 2. 构建 H^2EM 定义的难负样本集 H_i (同动词或同物品)
        mask_verb = (batch_verb.unsqueeze(1) == batch_verb.unsqueeze(0))
        mask_obj = (batch_obj.unsqueeze(1) == batch_obj.unsqueeze(0))
        # 难负样本：同动词或同物品，且必须排除对角线自身正样本
        mask_hard = (mask_verb | mask_obj) & ~torch.eye(B, dtype=torch.bool, device=v_hyp.device)
        
        # 3. Eq. 14 分母权重对齐: 在 logits 上加 ln(w) 等价于在 softmax 中乘 w
        if self.hard_weight > 1.0:
            penalty = math.log(self.hard_weight)
            logits[mask_hard] += penalty
            
        labels = torch.arange(B, device=v_hyp.device)
        
        # 4. 双向 InfoNCE
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2.0


# ==============================================================================
# 总损失函数计算路由
# ==============================================================================

def loss_calu(predict, target, config):
    """
    总损失入口
    """
    batch_img, batch_verb, batch_obj, batch_pair, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    
    c_pos = predict['c_pos']
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']
    
    v_hyp = predict['v_hyp']                  
    o_hyp = predict['o_hyp']                  
    t_v_hyp = predict['t_v_hyp']              
    t_o_hyp = predict['t_o_hyp']              
    coarse_v_hyp = predict['coarse_v_hyp']    
    coarse_o_hyp = predict['coarse_o_hyp']    

    ce_loss_fn = nn.CrossEntropyLoss()
    # 实例化 DAL，传入论文设定的难负样本权重 w=3.0
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07, hard_weight=3.0)
    # 实例化 HEM，常数 gamma (K) 为 0.1
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1)

    # 1. 基础分类损失 (严格对齐 Eq. 15 Primitive Auxiliary Loss)
    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)
    
    # 2. 判别对齐损失 (包含难负样本逻辑)
    loss_dal_verb = dal_loss_fn(v_hyp, t_v_hyp, c_pos, batch_verb, batch_obj)
    loss_dal_obj = dal_loss_fn(o_hyp, t_o_hyp, c_pos, batch_verb, batch_obj)
    loss_dal = loss_dal_verb + loss_dal_obj

    # 3. 层次蕴含损失 (四大偏序链，依据用户之前确认的跨模态层次方案)
    loss_hem_v2fv = hem_loss_fn(child=v_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_fv2cv = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_o2fo = hem_loss_fn(child=o_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_fo2co = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_v2fv + loss_hem_fv2cv + loss_hem_o2fo + loss_hem_fo2co

    # 4. 总损失汇总 (严格对齐 Eq. 16: L_total = beta1 * L_DA + beta2 * L_TE + beta3 * L_cls)
    w_cls = getattr(config, 'w_cls', 1.0)
    w_dal = getattr(config, 'w_dal', 1.0)
    w_hem = getattr(config, 'w_hem', 1.0)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem

    return total_loss

# ----------------- 保留的外部接口 -----------------
class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label, mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss * batch_size
        else:
            return loss

def hsic_loss(input1, input2, unbiased=False):
    pass

class Gml_loss(nn.Module):
    pass