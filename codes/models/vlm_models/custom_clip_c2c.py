import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

# 引入第一步部署的双曲基础算子
from utils.lorentz import expmap0 

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

        for block in self.transformer.resblocks:
            block.attn_mask = block.attn_mask[:cfg.ctx_length, :cfg.ctx_length]
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
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
        # Text Prompt Learners (Fine-grained)
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        # Encoders
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.token_embedding = clip_model.token_embedding  # 提取原始 embedding 层供粗粒度使用

        # ------------ 新增：粗粒度文本预处理与 Token化 ------------
        self.coarse_attrs = train_dataset.coarse_attrs
        self.coarse_objs = train_dataset.coarse_objs
        
        # 构造标准的 CLIP prompt
        coarse_verb_prompts = [f"a video of a person {c}" for c in self.coarse_attrs]
        coarse_obj_prompts = [f"a video of a {c}" for c in self.coarse_objs]
        
        # 将 Token 注册为 buffer，避免设备不对齐
        self.register_buffer('coarse_verb_tokens', clip.tokenize(coarse_verb_prompts))
        self.register_buffer('coarse_obj_tokens', clip.tokenize(coarse_obj_prompts))
        # ---------------------------------------------------------

        # Independent Learning Modules
        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        # ------------ 新增：双曲空间映射参数 ------------
        self.c = nn.Parameter(torch.tensor([1.0]))
        self.visual_scale = nn.Parameter(torch.tensor([0.1]))
        self.text_scale = nn.Parameter(torch.tensor([0.1]))
        # ------------------------------------------------

    # 修改前向传播签名，强制接收批次对应的文本标签索引
    def forward(self, video, batch_verb=None, batch_obj=None, batch_coarse_verb=None, batch_coarse_obj=None, pairs=None):
        device = video.device
        
        # 1. Fine-grained Text Features
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)

        # 2. Coarse-grained Text Features (新增数据流)
        # 用原生的 Token Embedding 获取输入，然后过定制的 text_encoder
        with torch.no_grad():
            c_v_emb = self.token_embedding(self.coarse_verb_tokens).type(self.text_encoder.dtype)
            c_o_emb = self.token_embedding(self.coarse_obj_tokens).type(self.text_encoder.dtype)
            
        coarse_verb_features = self.text_encoder(c_v_emb, self.coarse_verb_tokens)
        coarse_obj_features = self.text_encoder(c_o_emb, self.coarse_obj_tokens)
        
        # 保持空间一致性，过同一个线性映射层
        coarse_verb_features = self.c2c_text_v(coarse_verb_features)
        coarse_obj_features = self.c2c_text_o(coarse_obj_features)

        # 3. Video Features
        video_features = self.video_encoder(video)
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)

        # ------------ 严格的欧式到双曲 (Lorentz) 映射 ------------
        c_pos = F.softplus(self.c)

        # L2 归一化并缩放
        o_feat_norm = F.normalize(o_feat, p=2, dim=-1) * self.visual_scale
        v_feat_norm = F.normalize(v_feat, p=2, dim=-1) * self.visual_scale
        
        verb_text_norm = F.normalize(verb_text_features, p=2, dim=-1) * self.text_scale
        obj_text_norm = F.normalize(obj_text_features, p=2, dim=-1) * self.text_scale
        
        coarse_verb_norm = F.normalize(coarse_verb_features, p=2, dim=-1) * self.text_scale
        coarse_obj_norm = F.normalize(coarse_obj_features, p=2, dim=-1) * self.text_scale

        # 补 0 升维
        def to_lorentz_u(x):
            return torch.cat([torch.zeros(x.shape[0], 1, device=device), x], dim=-1)

        # 执行 expmap0
        o_hyp = expmap0(to_lorentz_u(o_feat_norm), c=c_pos)
        v_hyp = expmap0(to_lorentz_u(v_feat_norm), c=c_pos)
        
        t_v_hyp_all = expmap0(to_lorentz_u(verb_text_norm), c=c_pos)
        t_o_hyp_all = expmap0(to_lorentz_u(obj_text_norm), c=c_pos)
        
        coarse_v_hyp_all = expmap0(to_lorentz_u(coarse_verb_norm), c=c_pos)
        coarse_o_hyp_all = expmap0(to_lorentz_u(coarse_obj_norm), c=c_pos)

        # 4. 基础分类 Logits：洛伦兹内积与双曲距离
        def lorentz_inner(x, y):
            # x: [B, D+1], y: [N, D+1] -> return: [B, N]
            return -torch.matmul(x[:, 0:1], y[:, 0:1].t()) + torch.matmul(x[:, 1:], y[:, 1:].t())

        verb_inner = lorentz_inner(v_hyp, t_v_hyp_all)
        obj_inner = lorentz_inner(o_hyp, t_o_hyp_all)

        verb_dist = torch.acosh(torch.clamp(-verb_inner / c_pos, min=1.0 + 1e-5)) * torch.sqrt(c_pos)
        obj_dist = torch.acosh(torch.clamp(-obj_inner / c_pos, min=1.0 + 1e-5)) * torch.sqrt(c_pos)

        verb_logits = torch.exp(-verb_dist)
        obj_logits = torch.exp(-obj_dist)
        pred_com = torch.einsum('bi,bj->bij', verb_logits, obj_logits)

        # ------------ 训练与测试的返回分流 ------------
        if self.training:
            # 根据当前 batch 挑选对应的正样本对用于双曲蕴含锥计算
            t_v_hyp_batch = t_v_hyp_all[batch_verb]
            t_o_hyp_batch = t_o_hyp_all[batch_obj]
            coarse_v_hyp_batch = coarse_v_hyp_all[batch_coarse_verb]
            coarse_o_hyp_batch = coarse_o_hyp_all[batch_coarse_obj]

            # 必须返回字典以打通 loss_calu 数据流
            predict = {
                'c_pos': c_pos,
                'verb_logits': verb_logits,
                'obj_logits': obj_logits,
                'pred_com': pred_com,
                'v_hyp': v_hyp,
                'o_hyp': o_hyp,
                't_v_hyp': t_v_hyp_batch,
                't_o_hyp': t_o_hyp_batch,
                'coarse_v_hyp': coarse_v_hyp_batch,
                'coarse_o_hyp': coarse_o_hyp_batch
            }
            return predict
        else:
            verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
            com_logits = pred_com[:, verb_idx, obj_idx]
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
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    print("Building custom CLIP (Ablation Version)")
    model = CustomCLIP(cfg, train_dataset, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                if cfg.learn_input_method == 'coop':
                    if 'prompt_vectors' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'csp':
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'spm':
                    if 'prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                else:
                    raise NotImplementedError
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
                print(f'{name}: {param.requires_grad}')
        elif 'c2c' in name or name in ['c', 'visual_scale', 'text_scale']:
            param.requires_grad = True
            print(f'{name}: {param.requires_grad}')
    return model