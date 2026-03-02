import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# ==============================================================================
# 洛伦兹双曲空间基础几何算子 (严格对齐 H^2EM 与 Lorentz Model)
# ==============================================================================
def lorentz_inner(x, y):
    """洛伦兹内积 <x, y>_L"""
    return -x[:, 0:1] * y[:, 0:1] + (x[:, 1:] * y[:, 1:]).sum(dim=1, keepdim=True)

def origin_distance(x, c):
    """计算点到双曲原点 O=(1/sqrt(c), 0, ..., 0) 的双曲距离"""
    # d(O, x) = arccosh( sqrt(c) * x_0 ) / sqrt(c)
    return torch.acosh(torch.clamp(torch.sqrt(c) * x[:, 0:1], min=1.0 + 1e-5)) / torch.sqrt(c)

def spatial_angle(x, y):
    """计算两个洛伦兹点在空间维度上的夹角 (用于判断是否在蕴含锥内)"""
    x_bar = x[:, 1:]
    y_bar = y[:, 1:]
    cos_theta = F.cosine_similarity(x_bar, y_bar, dim=1).unsqueeze(1)
    return torch.acos(torch.clamp(cos_theta, -0.9999, 0.9999))

def cone_angle(x, c, K=0.1):
    """计算节点 x 的蕴含锥半角 (Aperture)"""
    # alpha(x) = arcsin( K / sinh(sqrt(c) * d(O, x)) )
    d_o = origin_distance(x, c)
    val = K / torch.clamp(torch.sinh(torch.sqrt(c) * d_o), min=1e-5)
    return torch.asin(torch.clamp(val, -0.9999, 0.9999))

# ==============================================================================
# H^2EM 核心损失函数实现
# ==============================================================================

class HierarchicalEntailmentLoss(nn.Module):
    """
    层次蕴含损失 (Entailment Cone Loss)
    严格约束 child 必须包含在 parent 的蕴含锥内，且 child 离原点更远（更具体）。
    """
    def __init__(self, K=0.1, margin=0.05):
        super().__init__()
        self.K = K
        self.margin = margin

    def forward(self, child, parent, c):
        # 1. 深度约束 (Depth Penalty): 越具体的概念(child)离原点应该越远
        d_child = origin_distance(child, c)
        d_parent = origin_distance(parent, c)
        loss_depth = F.relu(d_parent - d_child + self.margin)

        # 2. 锥形约束 (Cone Penalty): child 的空间方向必须落在 parent 的视角锥内
        theta = spatial_angle(child, parent)
        alpha_parent = cone_angle(parent, c, K=self.K)
        loss_cone = F.relu(theta - alpha_parent)

        return (loss_depth + loss_cone).mean()


class DiscriminativeAlignmentLoss(nn.Module):
    """
    判别对齐损失 (Discriminative Alignment Loss)
    在双曲空间中进行 InfoNCE 对比学习，拉近正样本对，排斥负样本对。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_hyp, t_hyp, c):
        # 计算全局 Lorentz 内积矩阵
        # v_hyp: [B, D+1], t_hyp: [N, D+1] -> inner: [B, N]
        inner = -torch.matmul(v_hyp[:, 0:1], t_hyp[:, 0:1].t()) + torch.matmul(v_hyp[:, 1:], t_hyp[:, 1:].t())
        
        # 转换为双曲距离矩阵
        dist = torch.acosh(torch.clamp(-inner / c, min=1.0 + 1e-5)) * torch.sqrt(c)
        
        # 将双曲距离转为对比学习的 logits (距离越近，相似度越高)
        logits = -dist / self.temperature
        
        # 构建对角线目标标签 (假设 v_hyp 和 t_hyp 是严格成对的 Batch)
        labels = torch.arange(v_hyp.size(0), device=v_hyp.device)
        
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2.0


# ==============================================================================
# 总损失函数计算路由
# ==============================================================================

def loss_calu(predict, target, config):
    """
    总损失计算
    输入要求：
    - target: 包含 6 个元素 (对应 Dataset 第二步修改)，分别为 [batch_img, batch_attr(verb), batch_obj, batch_pair, coarse_attr, coarse_obj]
    - predict: 必须是一个字典，包含计算双曲损失所需的各项特征和参数。
    """
    # 1. 解析目标标签
    batch_img, batch_verb, batch_obj, batch_pair, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    batch_pair = batch_pair.cuda()
    
    # 2. 解析预测字典 (要求模型 forward 返回该结构的字典)
    c_pos = predict['c_pos']
    
    # 分类 Logits (可保留欧式交叉熵，也可直接用距离，此处依基线保留 CE)
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']
    
    # 双曲特征 (Hyperbolic Embeddings)
    v_hyp = predict['v_hyp']                  # 视频特征
    o_hyp = predict['o_hyp']                  # 视频提取的物品特征
    t_v_hyp = predict['t_v_hyp']              # 细粒度动词特征
    t_o_hyp = predict['t_o_hyp']              # 细粒度物品特征
    
    coarse_v_hyp = predict['coarse_v_hyp']    # 粗粒度动词特征 (需在模型中补充提取)
    coarse_o_hyp = predict['coarse_o_hyp']    # 粗粒度物品特征 (需在模型中补充提取)

    # 初始化损失模块
    ce_loss_fn = CrossEntropyLoss()
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07).cuda()
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1, margin=0.05).cuda()

    # --------------------------------------------------------------------------
    # Part 1: 基础分类损失 (Verb & Object Classification Loss)
    # --------------------------------------------------------------------------
    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)
    
    # --------------------------------------------------------------------------
    # Part 2: 判别对齐损失 (Discriminative Alignment Loss)
    # --------------------------------------------------------------------------
    loss_dal_verb = dal_loss_fn(v_hyp, t_v_hyp, c_pos)
    loss_dal_obj = dal_loss_fn(o_hyp, t_o_hyp, c_pos)
    loss_dal = loss_dal_verb + loss_dal_obj

    # --------------------------------------------------------------------------
    # Part 3: 层次蕴含损失 (Hierarchical Entailment Loss - HEM)
    # 严格遵循四个偏序关系链：Video -> Fine -> Coarse
    # --------------------------------------------------------------------------
    # a. Video -> Fine-grained Verb
    loss_hem_v2fv = hem_loss_fn(child=v_hyp, parent=t_v_hyp, c=c_pos)
    # b. Fine-grained Verb -> Coarse-grained Verb
    loss_hem_fv2cv = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    
    # c. Video -> Fine-grained Object
    loss_hem_o2fo = hem_loss_fn(child=o_hyp, parent=t_o_hyp, c=c_pos)
    # d. Fine-grained Object -> Coarse-grained Object
    loss_hem_fo2co = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)

    loss_hem = loss_hem_v2fv + loss_hem_fv2cv + loss_hem_o2fo + loss_hem_fo2co

    # --------------------------------------------------------------------------
    # 总损失汇总 (权重系数应在 config 中定义，此处暂设默认比例)
    # --------------------------------------------------------------------------
    w_cls = getattr(config, 'w_cls', 1.0)
    w_dal = getattr(config, 'w_dal', 1.0)
    w_hem = getattr(config, 'w_hem', 1.0)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem

    return total_loss

# ----------------- 保留原有的辅助损失以防止其他依赖模块报错 -----------------
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
    # [保留原代码不动]
    pass

class Gml_loss(nn.Module):
    # [保留原代码不动]
    pass