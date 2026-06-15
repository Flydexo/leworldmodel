import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.linalg import vector_norm

class ActionEmbedder(nn.Module):
    def __init__(self, action_size, hidden_dim):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(action_size,hidden_dim),
            nn.SiLU()
        )
    def forward(self, x):
        return self.embed(x)

class Attention(nn.Module):
    def __init__(self, hidden_dim, nb_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, nb_heads, batch_first=True)

    def forward(self, x):
        return self.attention(x,x,x, need_weights=False)[0]

class CausalAttention(nn.Module):
    def __init__(self, hidden_dim, nb_heads, p_dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, nb_heads, batch_first=True, dropout=p_dropout)

    def forward(self, x):
        return self.attention(x,x,x, need_weights=False, is_causal=True, attn_mask=nn.modules.transformer.Transformer.generate_square_subsequent_mask(x.shape[-2]))[0]

class Patch(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B,C,H,W = x.shape
        P = self.patch_size
        N = H*W//(P**2)
        x = x.reshape(B,C,H//P,P,W//P,P)
        x = x.permute(0,2,4,1,3,5)
        x = x.reshape(B,N,-1)
        return x


class Predictor(nn.Module):
    def __init__(self, hidden_dim, nb_layers, p_dropout, nb_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition = nn.Linear(hidden_dim, 6*hidden_dim)
        with torch.no_grad():
            for p in self.condition.parameters():
                p.zero_()
        self.layers = nn.ModuleList([
             nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim,elementwise_affine=False),
                'att': CausalAttention(hidden_dim, nb_heads, p_dropout),
                'norm2': nn.LayerNorm(hidden_dim,elementwise_affine=False),
                'ffn': nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim), nn.GELU(), nn.Linear(4*hidden_dim, hidden_dim)),
             }) for _ in range(nb_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, action):
        # x.shape = B * T * D
        # action.shape = B * T * D
        alpha1,beta1,gamma1,alpha2,beta2,gamma2 = self.condition(action).chunk(6,-1)
        for layer in self.layers:
            x = self.dropout(layer['att'](layer['norm1'](x)*(1.0+gamma1)+beta1))*alpha1+x
            x = self.dropout(layer['ffn'](layer['norm2'](x)*(1.0+gamma2)+beta2))*alpha2+x
        return self.layer_norm(x)


class ViT(nn.Module):
    def __init__(self, hidden_dim, patch_size, channels, nb_heads, nb_layers, height, width):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.embedding = nn.Linear(patch_size**2*channels,hidden_dim)
        self.xclass = nn.Parameter(torch.randn(hidden_dim))
        n = height*width // (patch_size**2)
        self.pos_enc = nn.Parameter(torch.randn(n+1,hidden_dim))
        self.patch = Patch(patch_size)
        self.layers = nn.ModuleList([
             nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'att': Attention(hidden_dim, nb_heads),
                'norm2': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim), nn.GELU(), nn.Linear(4*hidden_dim, hidden_dim)),
             }) for _ in range(nb_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.last_layer_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        # x.shape = bs * c * height * weight
        p = self.patch_size
        c = self.channels
        n = x.shape[2]*x.shape[3]//(p**2)
        x = self.patch(x)
        x = torch.concat([self.xclass.expand(x.shape[0], 1, -1), self.embedding(x)], dim=-2)+self.pos_enc
        for layer in self.layers:
            x = layer['att'](layer['norm1'](x))+x
            x = layer['ffn'](layer['norm2'](x))+x
        return self.batch_norm(self.last_layer_mlp(self.layer_norm(x)[:,0,:]))


class WorldModel(nn.Module):
    def __init__(self, hidden_dim, patch_size, channels, enc_nb_heads, enc_nb_layers, height, width, pred_nb_layers, pred_p_dropout, pred_nb_heads, action_dim):
        super().__init__()
        self.encoder = ViT(hidden_dim, patch_size, channels, enc_nb_heads, enc_nb_layers, height, width)
        self.predictor = Predictor(hidden_dim, pred_nb_layers, pred_p_dropout, pred_nb_heads)
        self.action_embedder = ActionEmbedder(action_dim, hidden_dim)

    def forward(self, frames, actions):
        # frames.shape = B * T * C * H * W
        # actions.shape = B * T * action_dim
        actions = self.action_embedder(actions) # B * T * D
        frames = self.encode_frames(frames)
        predicted = self.predictor(frames, actions)
        return predicted, frames

    def encode_frames(self, frames):
        B,T,C,H,W = frames.shape
        frames = frames.reshape(-1, C, H, W) # (B*T)*C*H*W
        frames = self.encoder(frames)
        frames = frames.reshape(B,T,frames.shape[-1])
        return frames

    def rollout(self, start, actions, H):
        # start.shape = B * D
        # actions.shape = B * T * action_dim
        B, T, action_dim = actions.shape
        pred = None
        actions = self.action_embedder(actions)
        frames = [start.expand(B, 1, -1)]
        for i in range(H):
            t_frames = torch.cat(frames, dim=1)
            pred = self.predictor(t_frames, actions[:,0:i+1,:])
            frames.append(pred[:,-1:,:])
        return pred[:,-1,:]

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        _,_,_,H,W,C = info_dict['pixels'].shape
        device = next(self.parameters()).device
        info_dict['pixels'] = torch.nan_to_num(info_dict['pixels'], 0.0).to(device).type(torch.float32)
        info_dict['goal'] = torch.nan_to_num(info_dict['goal'] , 0.0).to(device).type(torch.float32)
        start = self.encode_frames(info_dict['pixels'].reshape(-1, H, W, C).transpose(1,-1).transpose(-1, -2).unsqueeze(1))
        E,N,T,AD = action_candidates.shape
        actions = action_candidates.reshape(-1,T,AD)
        destinations = self.rollout(start, actions, T)  # (N, D)
        goal = self.encode_frames(info_dict['goal'].view(-1, H, W, C).transpose(1,-1).transpose(-1, -2).unsqueeze(0))
        cost = vector_norm(goal - destinations.view(E, N, -1), dim=-1)
        
        return cost