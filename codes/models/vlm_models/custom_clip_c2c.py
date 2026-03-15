import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

# 严格对齐工具库
from utils.lorentz import exp_map0, log_map0, pairwise_dist

_tokenizer = _Tokenizer()

# =====================================================================
# 【防爆核心】：最大模长裁剪 (Norm Clipping)
# =====================================================================
def clip_by_norm(x, max_norm=3.0):
    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return x * scale

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)

class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.register_buffer('full_attn_mask', clip_model.transformer.resblocks[0].attn_mask.clone())
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
        seq_len = x.shape[0]
        for block in self.transformer.resblocks:
            block.attn_mask = self.full_attn_mask[:seq_len, :seq_len]

        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj
        self.num_frames = cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)
        return out

class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.token_embedding = clip_model.token_embedding

        self.coarse_attrs = train_dataset.coarse_attrs
        self.coarse_objs = train_dataset.coarse_objs
        coarse_verb_prompts = [f"a video of a person {c}" for c in self.coarse_attrs]
        coarse_obj_prompts = [f"a video of a {c}" for c in self.coarse_objs]
        self.register_buffer('coarse_verb_tokens', clip.tokenize(coarse_verb_prompts))
        self.register_buffer('coarse_obj_tokens', clip.tokenize(coarse_obj_prompts))

        self.comp_prompts = [f"a video of a person {v} {o}" for v, o in train_dataset.pairs]
        self.register_buffer('comp_tokens', clip.tokenize(self.comp_prompts))

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_CE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)

        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE2 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)

        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_c = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        self.c = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.visual_scale = nn.Parameter(torch.tensor([1.0]))
        self.text_scale = nn.Parameter(torch.tensor([1.0]))
        
        # 核心层级深度参数（初始化分别对应 粗、细、组合）
        self.d_coarse = nn.Parameter(torch.tensor([1.0]))
        self.d_fine = nn.Parameter(torch.tensor([2.0]))
        self.d_comp = nn.Parameter(torch.tensor([3.0]))

        self.cls_temp = nn.Parameter(torch.tensor([0.07]))
        self.use_hyperbolic = getattr(cfg, 'use_hyperbolic', True)

    # =====================================================================
    # 【严格对齐文献3】：Horosphere Projection (HP模块)
    # =====================================================================
    def horosphere_projection(self, hyp_x, curv):
        return log_map0(hyp_x, curv=curv)

    def condition_module_hyperbolic(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_v_e_o_euc = f_v_e_o.view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2))
        f_o_e_v_euc = f_o_e_v.view(b, n_v, c)

        return f_v_e_o_euc, f_o_e_v_euc

    def forward(self, video, batch_verb=None, batch_obj=None, batch_coarse_verb=None, batch_coarse_obj=None, pairs=None):
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)

        with torch.no_grad():
            c_v_emb = self.token_embedding(self.coarse_verb_tokens).type(self.text_encoder.dtype)
            c_o_emb = self.token_embedding(self.coarse_obj_tokens).type(self.text_encoder.dtype)

        coarse_verb_features = self.text_encoder(c_v_emb, self.coarse_verb_tokens)
        coarse_obj_features = self.text_encoder(c_o_emb, self.coarse_obj_tokens)
        coarse_verb_features = self.c2c_text_v(coarse_verb_features)
        coarse_obj_features = self.c2c_text_o(coarse_obj_features)

        video_features = self.video_encoder(video)

        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)
        v_c_feat = self.c2c_CE1(video_features.mean(dim=-1))

        o_feat_c = self.c2c_OE2(video_features.mean(dim=-1))
        v_feat_c = self.c2c_VE2(video_features).mean(dim=-1)

        c_pos = torch.clamp(F.softplus(self.c), min=0.5)

        with torch.cuda.amp.autocast(enabled=False):
            c_fp32 = c_pos.float()

            o_feat_fp32 = o_feat.float()
            v_feat_fp32 = v_feat.float()
            v_c_feat_fp32 = v_c_feat.float()

            verb_text_fp32 = verb_text_features.float()
            obj_text_fp32 = obj_text_features.float()

            b = video_features.shape[0]
            feat_dim_c = verb_text_fp32.shape[-1]
            n_v = verb_text_fp32.shape[0]
            n_o = obj_text_fp32.shape[0]

            if self.use_hyperbolic:
                # ==========================================
                # 终极修复区：深度注入 + HP 组合 + 概率对齐
                # ==========================================
                v_scale = torch.clamp(self.visual_scale.float(), min=0.01)
                t_scale = torch.clamp(self.text_scale.float(), min=0.01)

                # 利用 softplus 保证深度始终为正，且网络可以自发微调
                d_c = F.softplus(self.d_coarse)
                d_f = F.softplus(self.d_fine)
                d_m = F.softplus(self.d_comp)

                MAX_R = 3.5

                # 1. 注入层级深度 (核心修复！这保证了 H2EM 损失不塌陷)
                o_feat_scaled = o_feat_fp32 * v_scale * d_f
                v_feat_scaled = v_feat_fp32 * v_scale * d_f
                v_c_feat_scaled = v_c_feat_fp32 * v_scale * d_m

                o_cond_scaled = o_feat_c.float() * v_scale * d_f
                v_cond_scaled = v_feat_c.float() * v_scale * d_f

                verb_text_scaled = verb_text_fp32 * t_scale * d_f
                obj_text_scaled = obj_text_fp32 * t_scale * d_f
                coarse_verb_scaled = coarse_verb_features.float() * t_scale * d_c
                coarse_obj_scaled = coarse_obj_features.float() * t_scale * d_c

                # 防爆截断 (保留了相对长短)
                o_feat_clipped = clip_by_norm(o_feat_scaled, MAX_R)
                v_feat_clipped = clip_by_norm(v_feat_scaled, MAX_R)
                v_c_feat_clipped = clip_by_norm(v_c_feat_scaled, MAX_R)
                o_cond_clipped = clip_by_norm(o_cond_scaled, MAX_R)
                v_cond_clipped = clip_by_norm(v_cond_scaled, MAX_R)

                t_v_clipped = clip_by_norm(verb_text_scaled, MAX_R)
                t_o_clipped = clip_by_norm(obj_text_scaled, MAX_R)
                coarse_v_clipped = clip_by_norm(coarse_verb_scaled, MAX_R)
                coarse_o_clipped = clip_by_norm(coarse_obj_scaled, MAX_R)

                # 2. 【建立双曲概念空间】：统一映射到双曲流形
                o_hyp = exp_map0(o_feat_clipped, curv=c_fp32)
                v_hyp = exp_map0(v_feat_clipped, curv=c_fp32)
                v_c_hyp = exp_map0(v_c_feat_clipped, curv=c_fp32) 

                t_v_hyp_all = exp_map0(t_v_clipped, curv=c_fp32)
                t_o_hyp_all = exp_map0(t_o_clipped, curv=c_fp32)
                coarse_v_hyp_all = exp_map0(coarse_v_clipped, curv=c_fp32)
                coarse_o_hyp_all = exp_map0(coarse_o_clipped, curv=c_fp32)

                o_cond_hyp = exp_map0(o_cond_clipped, curv=c_fp32)
                v_cond_hyp = exp_map0(v_cond_clipped, curv=c_fp32)

                # 3. 【执行 HP 投影】：投影到极限球面（平坦切空间）
                v_cond_hp_flat = self.horosphere_projection(v_cond_hyp, curv=c_fp32)
                o_cond_hp_flat = self.horosphere_projection(o_cond_hyp, curv=c_fp32)
                t_v_hp_flat = self.horosphere_projection(t_v_hyp_all, curv=c_fp32)
                t_o_hp_flat = self.horosphere_projection(t_o_hyp_all, curv=c_fp32)

                # 4. C2C 原生条件融合
                cond_v_flat_out, cond_o_flat_out = self.condition_module_hyperbolic(
                    v_cond_hp_flat, o_cond_hp_flat, t_v_hp_flat, t_o_hp_flat, n_o, b, feat_dim_c, n_v
                )

                # 5. 重塑最深组合层级(d_m)，并重返双曲
                # (乘以 d_m / d_f 是因为融合后的特征模长继承了基元 d_f 的级别，需要拔高到组合级别)
                depth_ratio = (d_m / d_f).clamp(min=1.0)
                cond_o_flat_out = clip_by_norm(cond_o_flat_out * depth_ratio, MAX_R)
                cond_v_flat_out = clip_by_norm(cond_v_flat_out * depth_ratio, MAX_R)

                cond_o_hyp = exp_map0(cond_o_flat_out, curv=c_fp32) 
                cond_v_hyp = exp_map0(cond_v_flat_out, curv=c_fp32) 

                # --- 测地距离计算 ---
                verb_dist = pairwise_dist(v_hyp, t_v_hyp_all, curv=c_fp32)
                obj_dist = pairwise_dist(o_hyp, t_o_hyp_all, curv=c_fp32)
                cond_o_dist = pairwise_dist(cond_o_hyp.view(-1, feat_dim_c), t_o_hyp_all, curv=c_fp32).view(b, n_v, n_o)
                cond_v_dist = pairwise_dist(cond_v_hyp.view(-1, feat_dim_c), t_v_hyp_all, curv=c_fp32).view(b, n_o, n_v)

                temp = F.softplus(self.cls_temp) + 0.05
                verb_logits = -verb_dist / temp
                obj_logits = -obj_dist / temp
                cond_o_logits = -cond_o_dist / temp
                cond_v_logits = -cond_v_dist / temp

                # 动态/静态对数概率
                logits_dyn = cond_o_logits + verb_logits.unsqueeze(-1)  
                logits_sta = cond_v_logits.transpose(1, 2) + obj_logits.unsqueeze(1) 

                # 严格对齐 C2C 概率加法法则
                pred_com_logits = torch.logsumexp(torch.stack([logits_dyn, logits_sta], dim=0), dim=0)

                if self.training:
                    batch_comp_tokens = self.comp_tokens[pairs]
                    with torch.no_grad():
                        batch_comp_emb = self.token_embedding(batch_comp_tokens).type(self.text_encoder.dtype)
                    batch_comp_text_features = self.text_encoder(batch_comp_emb, batch_comp_tokens)
                    batch_comp_text_features = self.c2c_text_c(batch_comp_text_features)

                    with torch.cuda.amp.autocast(enabled=False):
                        # 工具人也赋予最深层级
                        t_c_feat_fp32 = clip_by_norm(batch_comp_text_features.float() * t_scale * d_m, MAX_R)
                        t_c_hyp_batch = exp_map0(t_c_feat_fp32, curv=c_fp32)

                    t_v_hyp_batch = t_v_hyp_all[batch_verb]
                    t_o_hyp_batch = t_o_hyp_all[batch_obj]
                    coarse_v_hyp_batch = coarse_v_hyp_all[batch_coarse_verb]
                    coarse_o_hyp_batch = coarse_o_hyp_all[batch_coarse_obj]

                    predict = {
                        'c_pos': c_pos, 'verb_logits': verb_logits, 'obj_logits': obj_logits,          
                        'pred_com_logits': pred_com_logits, 'v_hyp': v_hyp, 'o_hyp': o_hyp,
                        'v_c_hyp': v_c_hyp, 't_v_hyp': t_v_hyp_batch, 't_o_hyp': t_o_hyp_batch,
                        't_c_hyp': t_c_hyp_batch, 'coarse_v_hyp': coarse_v_hyp_batch, 'coarse_o_hyp': coarse_o_hyp_batch
                    }
                    return predict
                else:
                    verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                    return pred_com_logits[:, verb_idx, obj_idx]

            else:
                # ------------------------- 原生欧式分支 -------------------------
                cond_v_euc, cond_o_euc = self.condition_module_hyperbolic(
                    v_feat_c.float(), o_feat_c.float(), verb_text_fp32, obj_text_fp32, n_o, b, feat_dim_c, n_v
                )

                v_feat_n = F.normalize(v_feat_fp32, dim=-1)
                o_feat_n = F.normalize(o_feat_fp32, dim=-1)
                verb_text_n = F.normalize(verb_text_fp32, dim=-1)
                obj_text_n = F.normalize(obj_text_fp32, dim=-1)

                verb_logits = (v_feat_n @ verb_text_n.t()) * 0.5 + 0.5
                obj_logits = (o_feat_n @ obj_text_n.t()) * 0.5 + 0.5

                cond_v_n = F.normalize(cond_v_euc.float(), dim=-1) 
                cond_o_n = F.normalize(cond_o_euc.float(), dim=-1) 

                p_o_con_v = torch.einsum('bnc,mc->bnm', cond_o_n, obj_text_n) * 0.5 + 0.5
                p_v_con_o = torch.einsum('bnc,mc->bnm', cond_v_n, verb_text_n) * 0.5 + 0.5
                p_v_con_o = p_v_con_o.permute(0, 2, 1) 

                p_pair_v = p_o_con_v * verb_logits.unsqueeze(-1)
                p_pair_o = p_v_con_o * obj_logits.unsqueeze(1)
                pred_com_prob = p_pair_v + p_pair_o 

                if self.training:
                    return {'c_pos': c_pos, 'verb_logits': verb_logits, 'obj_logits': obj_logits, 'pred_com_prob': pred_com_prob}
                else:
                    verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                    return pred_com_prob[:, verb_idx, obj_idx]

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def build_model(train_dataset, cfg):
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    model = CustomCLIP(cfg, train_dataset, clip_model)
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                if cfg.learn_input_method == 'coop' and 'prompt_vectors' in name:
                    param.requires_grad_(True)
                elif cfg.learn_input_method in ['csp', 'spm']:
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name or 'prompt_vectors' in name:
                        param.requires_grad_(True)
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
        elif 'c2c' in name or name in ['visual_scale', 'text_scale', 'cls_temp', 'c', 'd_coarse', 'd_fine', 'd_comp']:
            param.requires_grad = True
    return model