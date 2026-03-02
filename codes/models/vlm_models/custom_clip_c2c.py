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
        self.logit_scale = clip_model.logit_scale
        self.token_embedding = clip_model.token_embedding 

        self.coarse_attrs = train_dataset.coarse_attrs
        self.coarse_objs = train_dataset.coarse_objs
        coarse_verb_prompts = [f"a video of a person {c}" for c in self.coarse_attrs]
        coarse_obj_prompts = [f"a video of a {c}" for c in self.coarse_objs]
        self.register_buffer('coarse_verb_tokens', clip.tokenize(coarse_verb_prompts))
        self.register_buffer('coarse_obj_tokens', clip.tokenize(coarse_obj_prompts))

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        self.c = nn.Parameter(torch.tensor([1.0]))
        self.visual_scale = nn.Parameter(torch.tensor([0.1]))
        self.text_scale = nn.Parameter(torch.tensor([0.1]))

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

        c_pos = F.softplus(self.c)

        o_feat_norm = F.normalize(o_feat, p=2, dim=-1) * self.visual_scale
        v_feat_norm = F.normalize(v_feat, p=2, dim=-1) * self.visual_scale
        verb_text_norm = F.normalize(verb_text_features, p=2, dim=-1) * self.text_scale
        obj_text_norm = F.normalize(obj_text_features, p=2, dim=-1) * self.text_scale
        coarse_verb_norm = F.normalize(coarse_verb_features, p=2, dim=-1) * self.text_scale
        coarse_obj_norm = F.normalize(coarse_obj_features, p=2, dim=-1) * self.text_scale

        o_hyp = exp_map0(o_feat_norm, curv=c_pos)
        v_hyp = exp_map0(v_feat_norm, curv=c_pos)
        t_v_hyp_all = exp_map0(verb_text_norm, curv=c_pos)
        t_o_hyp_all = exp_map0(obj_text_norm, curv=c_pos)
        coarse_v_hyp_all = exp_map0(coarse_verb_norm, curv=c_pos)
        coarse_o_hyp_all = exp_map0(coarse_obj_norm, curv=c_pos)

        verb_dist = pairwise_dist(v_hyp, t_v_hyp_all, curv=c_pos)
        obj_dist = pairwise_dist(o_hyp, t_o_hyp_all, curv=c_pos)

        # 【核心修正1】提取真正的未经 exp() 压扁的放大 Logits 用于交叉熵！
        # 使用 CLIP 自带的可学习温度系数 scale
        scale = self.logit_scale.exp()
        verb_logits = -verb_dist * scale
        obj_logits = -obj_dist * scale

        if self.training:
            t_v_hyp_batch = t_v_hyp_all[batch_verb]
            t_o_hyp_batch = t_o_hyp_all[batch_obj]
            coarse_v_hyp_batch = coarse_v_hyp_all[batch_coarse_verb]
            coarse_o_hyp_batch = coarse_o_hyp_all[batch_coarse_obj]

            predict = {
                'c_pos': c_pos,
                'verb_logits': verb_logits,
                'obj_logits': obj_logits,
                'v_hyp': v_hyp,
                'o_hyp': o_hyp,
                't_v_hyp': t_v_hyp_batch,
                't_o_hyp': t_o_hyp_batch,
                'coarse_v_hyp': coarse_v_hyp_batch,
                'coarse_o_hyp': coarse_o_hyp_batch
            }
            return predict
        else:
            # 【核心修正2】在测试时，使用 Softmax 把 logits 转成严格的概率
            # 这样 test.py 添加 bias=0.001 的逻辑才具有意义，并可做内积组合
            verb_prob = torch.softmax(verb_logits, dim=-1)
            obj_prob = torch.softmax(obj_logits, dim=-1)
            pred_com = torch.einsum('bi,bj->bij', verb_prob, obj_prob)
            
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
        elif 'c2c' in name or name in ['c', 'visual_scale', 'text_scale']:
            param.requires_grad = True
    return model