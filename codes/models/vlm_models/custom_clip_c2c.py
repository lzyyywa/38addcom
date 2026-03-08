import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

from utils.lorentz import exp_map0, pairwise_dist

_tokenizer = _Tokenizer()

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

        # 缓存组合 token
        comp_prompts = [f"a video of a person {v} {o}" for v, o in train_dataset.pairs]
        self.register_buffer('comp_tokens', clip.tokenize(comp_prompts))

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        # 独立特征提取网络
        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        
        # 提取全局组合视觉特征 v_c (作为正则化工具人)
        self.c2c_CE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)

        # 【新增：C2C 论文专属条件特征网络】
        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE2 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)

        # 【新增：C2C 论文欧式拼接融合层】
        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_c = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        self.c = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.visual_scale = nn.Parameter(torch.tensor([0.1]))
        self.text_scale = nn.Parameter(torch.tensor([0.1]))

        self.cls_temp = nn.Parameter(torch.tensor([0.07]))

    # 【新增方法：纯欧式拼接，供双曲映射使用】
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
        
        # 独立特征
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)
        
        # 工具人组合特征 v_c
        v_c_feat = self.c2c_CE1(video_features.mean(dim=-1))
        
        # 【新增：C2C 视觉条件特征】
        o_feat_c = self.c2c_OE2(video_features.mean(dim=-1))
        v_feat_c = self.c2c_VE2(video_features).mean(dim=-1)

        c_pos = torch.clamp(F.softplus(self.c), min=0.5)

        with torch.cuda.amp.autocast(enabled=False):
            c_fp32 = c_pos.float()
            
            # 基础特征转 fp32
            o_feat_fp32 = o_feat.float()
            v_feat_fp32 = v_feat.float()
            v_c_feat_fp32 = v_c_feat.float()
            
            verb_text_fp32 = verb_text_features.float()
            obj_text_fp32 = obj_text_features.float()
            
            # 【新增：调用欧式融合】
            b = video_features.shape[0]
            feat_dim_c = verb_text_fp32.shape[-1]
            n_v = verb_text_fp32.shape[0]
            n_o = obj_text_fp32.shape[0]
            
            # 在欧式空间拼接获得联合特征（无归一化）
            cond_v_euc, cond_o_euc = self.condition_module_hyperbolic(
                v_feat_c.float(), o_feat_c.float(), verb_text_fp32, obj_text_fp32, n_o, b, feat_dim_c, n_v
            )

            # --- 全员乘以缩放系数 (对齐尺度) ---
            o_feat_scaled = o_feat_fp32 * self.visual_scale.float()
            v_feat_scaled = v_feat_fp32 * self.visual_scale.float()
            v_c_feat_scaled = v_c_feat_fp32 * self.visual_scale.float()
            
            cond_v_scaled = cond_v_euc * self.visual_scale.float()
            cond_o_scaled = cond_o_euc * self.visual_scale.float()

            verb_text_scaled = verb_text_fp32 * self.text_scale.float()
            obj_text_scaled = obj_text_fp32 * self.text_scale.float()
            
            coarse_verb_scaled = coarse_verb_features.float() * self.text_scale.float()
            coarse_obj_scaled = coarse_obj_features.float() * self.text_scale.float()

            # --- 映射到双曲空间 (升维打击) ---
            o_hyp = exp_map0(o_feat_scaled, curv=c_fp32)
            v_hyp = exp_map0(v_feat_scaled, curv=c_fp32)
            v_c_hyp = exp_map0(v_c_feat_scaled, curv=c_fp32) # 工具人 vc 的双曲点

            t_v_hyp_all = exp_map0(verb_text_scaled, curv=c_fp32)
            t_o_hyp_all = exp_map0(obj_text_scaled, curv=c_fp32)
            coarse_v_hyp_all = exp_map0(coarse_verb_scaled, curv=c_fp32)
            coarse_o_hyp_all = exp_map0(coarse_obj_scaled, curv=c_fp32)
            
            # 【新增：条件特征的双曲点】
            cond_o_hyp = exp_map0(cond_o_scaled, curv=c_fp32) # [B, N_v, D]
            cond_v_hyp = exp_map0(cond_v_scaled, curv=c_fp32) # [B, N_o, D]

            # --- 计算双曲测地距离 ---
            verb_dist = pairwise_dist(v_hyp, t_v_hyp_all, curv=c_fp32)
            obj_dist = pairwise_dist(o_hyp, t_o_hyp_all, curv=c_fp32)
            
            # 【新增：条件特征计算距离，展平以适配 pairwise_dist】
            cond_o_dist = pairwise_dist(cond_o_hyp.view(-1, feat_dim_c), t_o_hyp_all, curv=c_fp32).view(b, n_v, n_o)
            cond_v_dist = pairwise_dist(cond_v_hyp.view(-1, feat_dim_c), t_v_hyp_all, curv=c_fp32).view(b, n_o, n_v)

            temp = F.softplus(self.cls_temp) + 0.05
            
            # --- 计算概率，严格复刻 C2C 公式！ ---
            verb_prob = torch.softmax(-verb_dist / temp, dim=-1)   # p(v)
            obj_prob = torch.softmax(-obj_dist / temp, dim=-1)     # p(o)
            
            p_o_con_v = torch.softmax(-cond_o_dist / temp, dim=-1) # p(o|v)
            p_v_con_o = torch.softmax(-cond_v_dist / temp, dim=-1) # p(v|o)
            
            # 动态路径: p(o|v) * p(v)
            p_pair_v = p_o_con_v * verb_prob.unsqueeze(-1)  # [B, N_v, N_o]
            # 静态路径: p(v|o) * p(o)
            p_pair_o = p_v_con_o.transpose(1, 2) * obj_prob.unsqueeze(1) # [B, N_v, N_o]
            
            # 双流联合，得到核心动作预测矩阵
            pred_com_prob = (p_pair_v + p_pair_o) / 2.0

        if self.training:
            # 工具人 t_c 提取，专门用于 HEM Loss
            batch_comp_tokens = self.comp_tokens[pairs]
            with torch.no_grad():
                batch_comp_emb = self.token_embedding(batch_comp_tokens).type(self.text_encoder.dtype)
            batch_comp_text_features = self.text_encoder(batch_comp_emb, batch_comp_tokens)
            batch_comp_text_features = self.c2c_text_c(batch_comp_text_features)
            
            with torch.cuda.amp.autocast(enabled=False):
                t_c_feat_fp32 = batch_comp_text_features.float() * self.text_scale.float()
                t_c_hyp_batch = exp_map0(t_c_feat_fp32, curv=c_fp32)

            t_v_hyp_batch = t_v_hyp_all[batch_verb]
            t_o_hyp_batch = t_o_hyp_all[batch_obj]
            coarse_v_hyp_batch = coarse_v_hyp_all[batch_coarse_verb]
            coarse_o_hyp_batch = coarse_o_hyp_all[batch_coarse_obj]

            predict = {
                'c_pos': c_pos,
                'verb_logits': -verb_dist / temp,   # 给基元算 CrossEntropy
                'obj_logits': -obj_dist / temp,     # 给基元算 CrossEntropy
                'pred_com_prob': pred_com_prob,     # 【核心】：送给 Loss 计算 L_com
                'v_hyp': v_hyp,
                'o_hyp': o_hyp,
                'v_c_hyp': v_c_hyp,                 # 工具人
                't_v_hyp': t_v_hyp_batch,
                't_o_hyp': t_o_hyp_batch,
                't_c_hyp': t_c_hyp_batch,           # 工具人
                'coarse_v_hyp': coarse_v_hyp_batch,
                'coarse_o_hyp': coarse_o_hyp_batch
            }
            return predict
        else:
            # 【完美闭环】：推断直接用 C2C 双流概率矩阵，彻底无视 v_c
            verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
            com_logits = pred_com_prob[:, verb_idx, obj_idx]
            return com_logits

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
        elif 'c2c' in name or name in ['visual_scale', 'text_scale', 'cls_temp','c']:
            param.requires_grad = True
    return model