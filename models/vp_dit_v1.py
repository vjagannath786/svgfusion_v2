# --- START OF FILE vp_dit.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm  # For progress bars
import os # Needed for saving models
import random
import sys # For sys.path.append
from transformers import AutoTokenizer, AutoModel # For CLIP in ddim_sample
import matplotlib.pyplot as plt

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Helper Modules ---


# --- Helper Functions ---
def modulate(x, shift, scale):
    """Apply AdaLN modulation with improved numerical stability."""
    return x * (1 + scale) + shift

def t2i_modulate(x, shift, scale):
    """Text-to-image style modulation."""
    return x * (1 + scale) + shift

def apply_rope(x):
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.
    Assumes x has shape [B, L, D].
    """
    bsz, seqlen, dim = x.shape
    if dim % 2 != 0: 
        padding = torch.zeros(bsz, seqlen, 1, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=-1)
        dim_padded = dim + 1
    else:
        x_padded = x
        dim_padded = dim
    half_dim = dim_padded // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=x.device); inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    pos = torch.arange(seqlen, dtype=torch.float32, device=x.device); sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos(); sin = sin[None, :, :].repeat(bsz, 1, 1); cos = cos[None, :, :].repeat(bsz, 1, 1)
    x1, x2 = x_padded[..., :half_dim], x_padded[..., half_dim:]; x_rotated_padded = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if dim % 2 != 0: return x_rotated_padded[..., :dim]
    return x_rotated_padded

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        t_device = t.device; half = dim // 2
        freqs = torch.exp( -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half ).to(t_device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(next(self.mlp.parameters()).device))
        return t_emb

class MLP(nn.Module):
    """A simple Feed Forward Network (MLP) block (using deeper version)."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act2 = act_layer()
        #self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act1(x); x = self.drop(x)
        x = self.fc2(x); x = self.act2(x); x = self.drop(x)
        #x = self.fc3(x); x = self.drop(x)
        return x

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, num_heads, kdim=None, vdim=None, batch_first=True):
        super().__init__(); 
        self.num_heads=num_heads; 
        self.d_model=d_model; 
        assert d_model % num_heads == 0; 
        self.batch_first=batch_first
        self.kdim = kdim if kdim is not None else d_model; 
        self.vdim = vdim if vdim is not None else d_model; 
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model); 
        self.w_k = nn.Linear(self.kdim, d_model); 
        self.w_v = nn.Linear(self.vdim, d_model); 
        self.w_o = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size=query.size(0); q=self.w_q(query); k=self.w_k(key); v=self.w_v(value); q=q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2); k=k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2); v=v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores=torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.d_k);
        if key_padding_mask is not None: scores=scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attn_mask is not None:
             if attn_mask.dim()==2: attn_mask=attn_mask.unsqueeze(0).unsqueeze(0)
             scores=scores.masked_fill(attn_mask, float('-inf'))
        attn=F.softmax(scores, dim=-1); context=torch.matmul(attn, v); context=context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model); output=self.w_o(context); return output

class SelfAttention(nn.Module):
    """Optimized self-attention with better numerical stability."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., fp32_attention=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fp32_attention = fp32_attention
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        
        # Use FP32 for attention computation if specified
        if self.fp32_attention:
            q, k = q.float(), k.float()
        
        # Handle autocast based on device
        device_type = x.device.type
        if device_type == 'mps':
            # MPS autocast support is limited/newer. 
            # If fp32_attention is True, we are already in FP32, so we can disable autocast context.
            context = torch.no_grad() # Dummy context
        else:
            context = torch.amp.autocast(device_type=device_type, enabled=not self.fp32_attention)

        with context:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                attn = attn.masked_fill(attn_mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class CrossAttention(nn.Module):
    """Efficient cross-attention for text conditioning."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., fp32_attention=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fp32_attention = fp32_attention
        
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Combined k,v for efficiency
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, context_mask=None):
        """
        x: [B, N, dim] - query (latent sequence)
        context: [B, S, dim] - key/value (text sequence)
        context_mask: [B, S] - padding mask for context (True = padding)
        """
        B, N, C = x.shape
        S = context.shape[1]
        
        # Query from latent sequence
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Key and Value from context sequence
        kv = self.kv_linear(context).reshape(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)  # [B, num_heads, S, head_dim]
        v = v.transpose(1, 2)  # [B, num_heads, S, head_dim]
        
        # Use FP32 for attention computation if specified
        if self.fp32_attention:
            q, k = q.float(), k.float()
        
        # Handle autocast based on device
        device_type = x.device.type
        if device_type == 'mps':
            # MPS autocast support is limited/newer. 
            # If fp32_attention is True, we are already in FP32, so we can disable autocast context.
            # If fp32_attention is False, we let it run in default precision (likely FP32 on MPS unless explicitly cast)
            context = torch.no_grad() # Dummy context
        else:
            context = torch.amp.autocast(device_type=device_type, enabled=not self.fp32_attention)

        with context:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, S]
            
            if context_mask is not None:
                # Expand mask: [B, S] -> [B, num_heads, N, S]
                mask = context_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, S)
                attn = attn.masked_fill(mask, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class FeedForward(nn.Module):
    """Improved feedforward network with GELU activation."""
    
    def __init__(self, dim, hidden_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim * 4
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --- VS-DiT Block (Modified for Sequence Context) ---

class VS_DiT_Block_vo(nn.Module):
    """ A VS-DiT block operating with internal hidden_dim, accepting sequence context. """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0): # context_dim removed here
        super().__init__()
        self.hidden_dim = hidden_dim; self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # AdaLN modulation is applied per-token in the sequence [B, N, D_hidden]
        modulation_dim = hidden_dim * 6 
        self.modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, modulation_dim, bias=True))
        
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, batch_first=True)
        # Cross attention key/value will come from the already projected context
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, kdim=hidden_dim, vdim=hidden_dim, batch_first=True)
        
        mlp_internal_dim = int(hidden_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = MLP(in_features=hidden_dim, hidden_features=mlp_internal_dim, out_features=hidden_dim, act_layer=approx_gelu, drop=dropout) # Pass dropout

    # MODIFIED forward signature: context_seq is now projected to hidden_dim
    def forward(self, x, t_emb, projected_context_seq, context_padding_mask=None):
        # x: [B, N, hidden_dim] (latent features)
        # t_emb: [B, hidden_dim] (timestep embedding)
        # projected_context_seq: [B, S_text, hidden_dim] (text features)
        # context_padding_mask: [B, S_text] (True where padded)

        # Compute modulation parameters from t_emb for AdaLN
        # gamma/beta are [B, hidden_dim], need to unsqueeze for [B, 1, hidden_dim] to broadcast over sequence length N
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.modulation_mlp(t_emb).chunk(6, dim=-1)
        gamma_1, beta_1, alpha_1 = gamma_1.unsqueeze(1), beta_1.unsqueeze(1), alpha_1.unsqueeze(1) # [B, 1, hidden_dim]
        gamma_2, beta_2, alpha_2 = gamma_2.unsqueeze(1), beta_2.unsqueeze(1), alpha_2.unsqueeze(1) # [B, 1, hidden_dim]

        # --- Self-Attention on latent (x) ---
        normed_x_sa = self.norm1(x) # x: [B, N, hidden_dim]
        #normed_x_sa = x
        
        mod_x_sa = normed_x_sa * (1 + gamma_1) + beta_1 # Apply AdaLN modulation
        sa_out = self.self_attn(mod_x_sa, mod_x_sa, mod_x_sa) # Query, Key, Value are all from x
        x = x + alpha_1 * sa_out # Residual update

        # --- Cross-Attention (Query: latent, Key/Value: projected_context_seq) ---
        normed_x_ca = self.norm2(x) # x: [B, N, hidden_dim]
        #3normed_x_ca = x
        # Modulation for cross-attention's query (often done after norm)
        # mod_x_ca = normed_x_ca * (1 + gamma_2) + beta_2 # Applying mod here or after CA depends on DiT variant
        ca_out = self.cross_attn(
            query=normed_x_ca, # Latent sequence is query [B, N, hidden_dim]
            key=projected_context_seq, # Projected context is key [B, S_text, hidden_dim]
            value=projected_context_seq, # Projected context is value [B, S_text, hidden_dim]
            key_padding_mask=context_padding_mask # Mask for context sequence
        )
        x = x + ca_out # Residual update

        # --- FeedForward on latent (x) ---
        normed_x_ff = self.norm3(x) # x: [B, N, hidden_dim]
        #normed_x_ff = x
        mod_x_ff = normed_x_ff * (1 + gamma_2) + beta_2 # Apply AdaLN modulation
        ff_out = self.mlp(mod_x_ff)
        x = x + alpha_2 * ff_out # Residual update

        return x # Output: [B, N, hidden_dim]

class VS_DiT_Block(nn.Module):
    """Improved VS-DiT block with better numerical stability and modulation."""
    
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0, fp32_attention=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fp32_attention = fp32_attention
        
        # Layer norms (without learnable parameters for AdaLN)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # AdaLN modulation MLPs
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
        # Attention layers
        self.self_attn = SelfAttention(
            hidden_dim, num_heads=num_heads, 
            attn_drop=dropout, proj_drop=dropout,
            fp32_attention=fp32_attention
        )
        
        self.cross_attn = CrossAttention(
            hidden_dim, num_heads=num_heads,
            attn_drop=dropout, proj_drop=dropout,
            fp32_attention=fp32_attention
        )
        
        # Feedforward network
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = FeedForward(
            hidden_dim, hidden_dim=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout
        )

    def forward(self, x, t_emb, context, context_mask=None, latent_mask=None):
        """
        x: [B, N, hidden_dim] - latent sequence
        t_emb: [B, hidden_dim] - timestep embedding
        context: [B, S, hidden_dim] - text context (already projected)
        context_mask: [B, S] - padding mask for context (True = padding)
        latent_mask: [B, N] - padding mask for latents (True = padding)
        """
        B, N, D = x.shape
        
        # Get modulation parameters
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Reshape for broadcasting: [B, D] -> [B, 1, D]
        shift_sa = shift_sa.unsqueeze(1)
        scale_sa = scale_sa.unsqueeze(1)
        gate_sa = gate_sa.unsqueeze(1)
        shift_ff = shift_ff.unsqueeze(1)
        scale_ff = scale_ff.unsqueeze(1)
        gate_ff = gate_ff.unsqueeze(1)
        
        # Self-attention branch
        x_norm1 = self.norm1(x)
        x_modulated1 = modulate(x_norm1, shift_sa, scale_sa)
        x_sa = self.self_attn(x_modulated1, attn_mask=latent_mask) # Pass latent_mask to self_attn
        x = x + gate_sa * x_sa
        
        # Cross-attention branch
        x_norm2 = self.norm2(x)
        #
        x_ca = self.cross_attn(x_norm2, context, context_mask)
        x = x +  x_ca
        
        # Feedforward branch
        x_norm3 = self.norm3(x)
        x_modulated3 = modulate(x_norm3, shift_ff, scale_ff)
        x_ff = self.mlp(x_modulated3)
        x = x + gate_ff * x_ff
        
        return x

# --- VS-DiT Model ---

class VS_DiT(nn.Module):
    """Improved VS-DiT model with better initialization and stability."""
    
    def __init__(self, latent_dim, hidden_dim, context_dim, num_blocks, num_heads, 
                 mlp_ratio=4.0, dropout=0.1, fp32_attention=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.fp32_attention = fp32_attention
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_dim)
        
        # Input projection
        self.proj_in = nn.Linear(latent_dim, hidden_dim, bias=True)
        
        # Context projection
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            VS_DiT_Block(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                fp32_attention=fp32_attention
            ) for _ in range(num_blocks)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim, bias=True)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Proper weight initialization for stable training."""
        # Initialize all linear layers with Xavier uniform
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Special initialization for specific layers
        
        # Timestep embedder
        if hasattr(self.t_embedder, 'mlp'):
            for layer in self.t_embedder.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Zero-out the output of adaLN modulation MLPs
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final modulation and output projection
        nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

    def forward(self, z_t, t, context_seq, context_padding_mask=None, latent_mask=None):
        """
        z_t: [B, N, latent_dim] - noisy latent sequence
        t: [B] - timestep
        context_seq: [B, S, context_dim] - text embedding sequence
        context_padding_mask: [B, S] - padding mask for context
        """
        device = next(self.parameters()).device
        B, N, _ = z_t.shape
        
        # Move inputs to device
        z_t = z_t.to(device)
        t = t.to(device)
        context_seq = context_seq.to(device)
        if context_padding_mask is not None:
            context_padding_mask = context_padding_mask.to(device)
        
        # Add positional embeddings
        pos_embed = get_sinusoidal_pos_embed(N, self.latent_dim, device)
        z_t = z_t + pos_embed.unsqueeze(0)
        
        # Get timestep embedding
        t_emb = self.t_embedder(t)  # [B, hidden_dim]
        
        # Project inputs to hidden dimension
        h = self.proj_in(z_t)  # [B, N, hidden_dim]
        context = self.context_proj(context_seq)  # [B, S, hidden_dim]
        
        # Apply DiT blocks
        for block in self.blocks:
            h = block(h, t_emb, context, context_padding_mask,latent_mask)
        
        # Final processing
        shift, scale = self.final_adaLN_modulation(t_emb).chunk(2, dim=1)
        shift = shift.unsqueeze(1)  # [B, 1, hidden_dim]
        scale = scale.unsqueeze(1)  # [B, 1, hidden_dim]
        
        h = self.final_norm(h)
        h = modulate(h, shift, scale)
        epsilon_pred = self.final_proj(h)  # [B, N, latent_dim]
        
        return epsilon_pred
    
    def forward_with_cfg(self, z_t, t, context_seq, context_padding_mask, cfg_scale, 
                        clip_model, clip_tokenizer):
        """
        Optimized forward pass with batched classifier-free guidance.
        """
        batch_size = z_t.shape[0]
        device = z_t.device
        
        # Create unconditional context
        text_seq_len = context_seq.size(1)
        empty_text_inputs = clip_tokenizer(
            [""] * batch_size,
            padding='max_length',
            max_length=text_seq_len,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            empty_outputs = clip_model(**empty_text_inputs)
            uncond_context = empty_outputs.last_hidden_state
            uncond_mask = ~(empty_text_inputs.attention_mask.bool())
        
        # Batch conditional and unconditional
        z_t_combined = torch.cat([z_t, z_t], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        context_combined = torch.cat([context_seq, uncond_context], dim=0)
        mask_combined = torch.cat([context_padding_mask, uncond_mask], dim=0)
        
        # Single forward pass
        eps_combined = self.forward(z_t_combined, t_combined, context_combined, mask_combined)
        
        # Split and apply guidance
        eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        
        return eps_pred, eps_cond, eps_uncond


# --- VS-DiT Model (Modified for Sequence Context) ---

def get_sinusoidal_pos_embed(seq_len, dim, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [seq_len, dim]

class VS_DiT_vo(nn.Module):
    """ VS-DiT model accepting sequence context. """
    def __init__(self, latent_dim, hidden_dim, context_dim, num_blocks, num_heads, mlp_ratio=8.0, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim; self.hidden_dim = hidden_dim; self.num_blocks = num_blocks
        
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_dim)
        self.proj_in = nn.Linear(latent_dim, hidden_dim) # Maps z_t from latent_dim to hidden_dim
        
        # --- MODIFIED: Context projection moved to main model ---
        # Projects text context from its original CLIP dimension (context_dim) to the model's hidden_dim
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        
        self.blocks = nn.ModuleList([
            VS_DiT_Block(
                hidden_dim=hidden_dim, num_heads=num_heads, # context_dim removed here
                mlp_ratio=mlp_ratio, dropout=dropout
            ) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True) # Outputs final gamma/beta
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim, bias=True) # Maps back to latent_dim (for epsilon)

        #self.initialize_weights()

    def initialize_weights(self):
        # Basic initialization for all linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Input projection (latent to hidden)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        
        # Context projection (text to hidden) - Xavier for general transformation
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        # Timestep embedding MLP
        if hasattr(self.t_embedder, 'mlp'):
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.t_embedder.mlp[0].bias)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.t_embedder.mlp[2].bias)

        # Zero-out output of modulation MLPs in DiT blocks
        for block in self.blocks:
            if hasattr(block, 'modulation_mlp') and isinstance(block.modulation_mlp[-1], nn.Linear):
                nn.init.constant_(block.modulation_mlp[-1].weight, 0)
                nn.init.constant_(block.modulation_mlp[-1].bias, 0)

        # Zero-out final modulation MLP and final output projection
        if isinstance(self.final_modulation_mlp[-1], nn.Linear):
            nn.init.constant_(self.final_modulation_mlp[-1].weight, 0)
            nn.init.constant_(self.final_modulation_mlp[-1].bias, 0)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)
    
    # MODIFIED forward signature: z_t is now [B, N, latent_dim]
    def forward(self, z_t, t, context_seq, context_padding_mask=None, latent_mask=None):
        # z_t: [B, N, latent_dim] (noisy latent representation)
        # t: [B] (timestep)
        # context_seq: [B, S_text, context_dim] (text embedding sequence)
        # context_padding_mask: [B, S_text] (True where padded in text context)
        # latent_mask: [B, N] (True where padded in latent sequence)

        model_device = next(self.parameters()).device
        
        # --- Add sinusoidal positional embeddings to z_t before projection ---
        # pos_embed: [N, latent_dim]
        pos_embed = get_sinusoidal_pos_embed(z_t.size(1), z_t.size(2), z_t.device) 
        z_t = z_t.to(model_device) + pos_embed.unsqueeze(0) # Add batch dim to pos_embed: [1, N, latent_dim]

        # --- Apply RoPE for sequential dependencies after adding fixed positional embeddings ---
        #z_t = apply_rope(z_t) # z_t is [B, N, latent_dim]

        t = t.to(model_device)
        context_seq = context_seq.to(model_device)
        if context_padding_mask is not None:
            context_padding_mask = context_padding_mask.to(model_device)
        if latent_mask is not None:
            latent_mask = latent_mask.to(model_device)

        t_emb = self.t_embedder(t) # Output: [B, hidden_dim]
        
        h = self.proj_in(z_t)      # Output: [B, N, hidden_dim] - latent projected to hidden_dim
        
        # --- Project the context sequence once ---
        projected_context_seq = self.context_proj(context_seq) # Output: [B, S_text, hidden_dim]

        # Pass h (latent sequence), t_emb, and projected_context_seq (text sequence) to blocks
        for block in self.blocks:
            h = block(h, t_emb, projected_context_seq, context_padding_mask, latent_mask) # h remains [B, N, hidden_dim]

        # Final modulation and projection
        final_gamma, final_beta = self.final_modulation_mlp(t_emb).chunk(2, dim=-1)
        final_gamma, final_beta = final_gamma.unsqueeze(1), final_beta.unsqueeze(1) # [B, 1, hidden_dim] to broadcast over N

        normed_h = self.final_norm(h) # h: [B, N, hidden_dim]
        mod_h = normed_h * (1 + final_gamma) + final_beta # Apply AdaLN modulation
        epsilon_pred = self.final_proj(mod_h) # Output: [B, N, latent_dim] - predicted noise in latent space
        
        return epsilon_pred




# --- Diffusion Utilities ---

def get_linear_noise_schedule(timesteps):
    beta_start = 0.0001; beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def precompute_diffusion_parameters(betas, target_device):
    betas = betas.to(target_device); alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(target_device)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=target_device), alphas_cumprod[:-1]])
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(target_device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(target_device)
    return { "betas": betas, "alphas_cumprod": alphas_cumprod,
             "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
             "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
             "alphas_cumprod_prev": alphas_cumprod_prev }

def noise_latent(z0, t, diff_params, target_device):
    # z0: [B, N, latent_dim]
    # t: [B]
    z0 = z0.to(target_device); t = t.to(target_device)
    # Ensure correct broadcasting: view(-1, 1, 1) to match [B, N, latent_dim]
    sqrt_alpha_bar = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1)
    sqrt_one_minus_alpha_bar = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1)
    epsilon = torch.randn_like(z0, device=target_device) # epsilon also [B, N, latent_dim]
    zt = sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * epsilon
    return zt, epsilon

# --- Sampling (DDIM) ---

@torch.no_grad()
def ddim_sample(model, shape, context_seq, context_padding_mask,
                diff_params, num_timesteps, target_device, cfg_scale=3.0, eta=0.0,
                clip_model=None, clip_tokenizer=None, return_visuals=False, # Changed default to False
                latent_min_max=(-14, 14)): # Adjusted clamp range based on discussion

    model.eval()
    batch_size = shape[0]
    z_t = torch.randn(shape, device=target_device) # Initial noise: [B, N, Dz]

    # Move context to device
    context_seq = context_seq.to(target_device) # [B, S_text, D_clip]
    if context_padding_mask is not None:
        context_padding_mask = context_padding_mask.to(target_device) # [B, S_text]

    if clip_model is None or clip_tokenizer is None:
        raise ValueError("clip_model and clip_tokenizer must be provided for ddim_sample.")

    # Get empty string embedding for unconditional context
    text_seq_len = context_seq.size(1) # Get S_text from conditional context
    empty_text_inputs = clip_tokenizer(
        [""] * batch_size, # Create a batch of empty strings
        padding='max_length',
        max_length=text_seq_len,
        truncation=True,
        return_tensors="pt"
    ).to(target_device)

    empty_outputs = clip_model(**empty_text_inputs)
    uncond_context_seq = empty_outputs.last_hidden_state # Already [B, S_text, D_clip]
    uncond_padding_mask = ~(empty_text_inputs.attention_mask.bool()) # Already [B, S_text]

    # Track values for plotting
    cosine_sims = []
    cfg_scales_used = []

    for i in tqdm(reversed(range(num_timesteps)), desc="DDIM Sampling (Sequential Latent)"):
        t = torch.full((batch_size,), i, dtype=torch.long, device=target_device)

        # Ensure correct broadcasting for alpha/sigma terms
        alpha_bar_t_val = diff_params["alphas_cumprod"][t].view(-1, 1, 1) # [B, 1, 1]
        alpha_bar_t_prev_val = diff_params["alphas_cumprod_prev"][t].view(-1, 1, 1) # [B, 1, 1]
        sqrt_one_minus_alpha_bar_t_val = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1) # [B, 1, 1]
        sqrt_alpha_bar_t_val = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1) # [B, 1, 1]

        # Model expects z_t: [B, N, Dz], t: [B], context_seq: [B, S_text, D_clip], context_padding_mask: [B, S_text]
        # Model output eps_cond/eps_uncond should be [B, N, Dz]
        eps_cond = model(z_t, t, context_seq, context_padding_mask)
        eps_uncond = model(z_t, t, uncond_context_seq, uncond_padding_mask)

        # Cosine similarity: flatten over sequence and feature dimensions
        cosine_sim = F.cosine_similarity(eps_cond.reshape(batch_size, -1),
                                         eps_uncond.reshape(batch_size, -1),
                                         dim=1).mean().item()
        cosine_sims.append(cosine_sim)

        # Dynamic CFG scale (logic unchanged, but now uses config['cfg_scale'])
        # min_sim = 0.90
        # max_sim = 0.99
        # cosine_sim_clamped = max(min(cosine_sim, max_sim), min_sim)
        # scale_factor = 1 + ((cosine_sim_clamped - min_sim) / (max_sim - min_sim))
        # current_cfg = cfg_scale * scale_factor
        # current_cfg = min(current_cfg, cfg_scale * 2.0) # Cap the dynamic scale
        # cfg_scales_used.append(current_cfg)

        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond) # Use current_cfg for scaling

        # Sigma_t calculation
        term_in_sqrt = (1.0 - alpha_bar_t_prev_val) / (1.0 - alpha_bar_t_val + 1e-12) * \
                       (1.0 - alpha_bar_t_val / (alpha_bar_t_prev_val + 1e-12))
        sigma_t = eta * torch.sqrt(torch.clamp(term_in_sqrt, min=1e-12)) # [B,1,1]

        # Predict x0
        pred_x0 = (z_t - sqrt_one_minus_alpha_bar_t_val * eps_pred) / (sqrt_alpha_bar_t_val + 1e-12)
        if latent_min_max is not None:
            pred_x0 = torch.clamp(pred_x0, latent_min_max[0], latent_min_max[1]) # [B, N, Dz]

        # Deterministic part for x_t_prev
        sqrt_term_for_dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_t_prev_val - sigma_t**2, min=0.0))
        dir_xt = sqrt_term_for_dir_xt * eps_pred # [B, N, Dz]

        x_t_prev = torch.sqrt(alpha_bar_t_prev_val) * pred_x0 + dir_xt # [B, N, Dz]

        if eta > 0: # Add noise if eta > 0
            noise_vec = torch.randn_like(z_t) # [B, N, Dz]
            x_t_prev += sigma_t * noise_vec # [B, N, Dz]

        z_t = x_t_prev

    # Plotting (logic unchanged, but uses cfg_scales_used)
    if return_visuals:
        steps = list(range(num_timesteps))[::-1]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(steps, cosine_sims, label='Cosine Similarity', color='blue')
        ax1.set_ylabel('Cosine Similarity'); ax1.set_xlabel('Timestep'); ax1.set_title('Cosine Similarity Over Time'); ax1.grid(True); ax1.invert_xaxis()
        ax2.plot(steps, cfg_scales_used, label='CFG Scale', color='green')
        ax2.set_ylabel('CFG Scale'); ax2.set_xlabel('Timestep'); ax2.set_title('CFG Scale Over Time'); ax2.grid(True); ax2.invert_xaxis()
        plt.tight_layout(); 
        plt.savefig('tmp_ddim_sampling_stats.png') # Changed filename
        plt.show()

    return z_t, {"cosine_sims": cosine_sims, "cfg_scales": cfg_scales_used}


@torch.no_grad()
def ddim_sample_fixed(model, shape, context_seq, context_padding_mask,
                     diff_params, num_timesteps, target_device, cfg_scale=3.0, eta=0.0,
                     clip_model=None, clip_tokenizer=None, return_visuals=False,
                     latent_min_max=(-10, 10), ddim_steps=100, latent_mask=None): # Added latent_mask
    """
    Fixed DDIM sampling with proper mathematics and batched CFG.
    """
    model.eval()
    batch_size = shape[0]
    
    # Initialize with noise
    x_t = torch.randn(shape, device=target_device)
    
    # Move context to device
    context_seq = context_seq.to(target_device)
    if context_padding_mask is not None:
        context_padding_mask = context_padding_mask.to(target_device)
    if latent_mask is not None:
        latent_mask = latent_mask.to(target_device)

    if clip_model is None or clip_tokenizer is None:
        raise ValueError("clip_model and clip_tokenizer must be provided for ddim_sample.")

    # Create timestep schedule for DDIM (subset of training steps)
    if ddim_steps < num_timesteps:
        # Use a subset of timesteps for faster sampling
        timesteps = torch.linspace(num_timesteps - 1, 0, ddim_steps).long()
    else:
        timesteps = torch.arange(num_timesteps - 1, -1, -1)
    
    # Pre-compute unconditional context
    text_seq_len = context_seq.size(1)
    empty_text_inputs = clip_tokenizer(
        [""] * batch_size,
        padding='max_length',
        max_length=text_seq_len,
        truncation=True,
        return_tensors="pt"
    ).to(target_device)

    with torch.no_grad():
        empty_outputs = clip_model(**empty_text_inputs)
        uncond_context_seq = empty_outputs.last_hidden_state
        uncond_padding_mask = ~(empty_text_inputs.attention_mask.bool())

    # Track diagnostics
    cosine_sims = []
    pred_x0_norms = []

    for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
        t_batch = t.repeat(batch_size).to(target_device)
        
        # Get diffusion parameters for current timestep
        alpha_t = diff_params["alphas_cumprod"][t].item()
        
        # Get next timestep parameters
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t_next = diff_params["alphas_cumprod"][t_next].item()
        else:
            alpha_t_next = 1.0  # At t=0, alpha_cumprod = 1
        
        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = math.sqrt(1 - alpha_t)
        sqrt_alpha_t_next = math.sqrt(alpha_t_next)
        sqrt_one_minus_alpha_t_next = math.sqrt(1 - alpha_t_next)

        # Classifier-Free Guidance with batched inference
        if cfg_scale > 1.0:
            # Batch conditional and unconditional
            x_t_combined = torch.cat([x_t, x_t], dim=0)
            t_combined = torch.cat([t_batch, t_batch], dim=0)
            context_combined = torch.cat([context_seq, uncond_context_seq], dim=0)
            mask_combined = torch.cat([context_padding_mask, uncond_padding_mask], dim=0)
            
            # Handle latent mask for batching
            latent_mask_combined = None
            if latent_mask is not None:
                latent_mask_combined = torch.cat([latent_mask, latent_mask], dim=0)

            # Single forward pass
            eps_combined = model(x_t_combined, t_combined, context_combined, mask_combined, latent_mask=latent_mask_combined)
            
            # Split results
            eps_cond, eps_uncond = eps_combined.chunk(2, dim=0)
            
            # Apply guidance
            eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            # Compute cosine similarity for diagnostics
            cosine_sim = F.cosine_similarity(
                eps_cond.reshape(batch_size, -1),
                eps_uncond.reshape(batch_size, -1),
                dim=1
            ).mean().item()
            cosine_sims.append(cosine_sim)
        else:
            # No guidance
            eps_pred = model(x_t, t_batch, context_seq, context_padding_mask, latent_mask=latent_mask)
            cosine_sims.append(0.0)

        # Predict x_0 using the reparameterization
        pred_x0 = (x_t - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t
        
        # Clamp predicted x_0 if specified
        if latent_min_max is not None:
            pred_x0 = torch.clamp(pred_x0, latent_min_max[0], latent_min_max[1])
        
        pred_x0_norms.append(pred_x0.norm().item())

        # Compute sigma for DDIM
        if eta > 0 and i < len(timesteps) - 1:
            sigma = eta * math.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * math.sqrt(1 - alpha_t / alpha_t_next)
        else:
            sigma = 0.0

        # Compute direction pointing towards x_t
        if i < len(timesteps) - 1:
            direction = math.sqrt(1 - alpha_t_next - sigma**2) * eps_pred
        else:
            direction = torch.zeros_like(eps_pred)

        # Compute x_{t-1}
        x_t = sqrt_alpha_t_next * pred_x0 + direction
        
        # Add noise if eta > 0
        if sigma > 0:
            noise = torch.randn_like(x_t)
            x_t = x_t + sigma * noise

    # Plotting diagnostics
    if return_visuals and cosine_sims:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        steps = range(len(cosine_sims))
        ax1.plot(steps, cosine_sims, 'b-', linewidth=2)
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_xlabel('DDIM Step')
        ax1.set_title('Conditional vs Unconditional Cosine Similarity')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(steps, pred_x0_norms, 'r-', linewidth=2)
        ax2.set_ylabel('Predicted x_0 Norm')
        ax2.set_xlabel('DDIM Step')
        ax2.set_title('Predicted x_0 Magnitude Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ddim_sampling_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.show()

    return x_t, {
        "cosine_sims": cosine_sims,
        "pred_x0_norms": pred_x0_norms,
        "final_norm": x_t.norm().item()
    }


# --- Training (Modified for Sequence Context Simulation/Eval) ---

def train(model, train_steps, batch_size, latent_dim, num_svg_tokens, context_dim, # num_svg_tokens added
          num_timesteps, cfg_prob, log_interval, eval_interval, cfg_scale_eval, target_device,
          model_save_dir="saved_models", max_text_seq_len=77): # Renamed max_seq_len to max_text_seq_len

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    betas_cpu = get_linear_noise_schedule(num_timesteps)
    diff_params = precompute_diffusion_parameters(betas_cpu, target_device)
    model.to(target_device); model.train()
    best_eval_loss = float('inf')
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "vs_dit_seqcond_best.pth")
    global_step = 0
    pbar = tqdm(total=train_steps, desc="Training")

    # Load CLIP for simulating context for DDIM
    print("Loading CLIP model for DDIM sampling during training simulation...")
    clip_model_name = "openai/clip-vit-base-patch32" # Example CLIP model
    clip_tokenizer_sim = AutoTokenizer.from_pretrained(clip_model_name)
    clip_model_sim = AutoModel.from_pretrained(clip_model_name).to(target_device)
    clip_model_sim.eval()


    while global_step < train_steps:
        # --- Simulate a batch of data ---
        # z0_batch is now [B, N, latent_dim]
        z0_batch_cpu = torch.randn(batch_size, num_svg_tokens, latent_dim) * 2.0 - 1.0 # Simulate some spread
        
        # Simulate sequence context and mask from CLIP
        # Max text sequence length (e.g., 77 for CLIP)
        sim_text_seq_len = max_text_seq_len
        # Simulate text embeddings: [B, S_text, D_clip]
        context_seq_cpu = torch.randn(batch_size, sim_text_seq_len, context_dim) 
        # Simulate padding mask (True where padded)
        context_padding_mask_cpu = torch.zeros(batch_size, sim_text_seq_len, dtype=torch.bool)

        z0_batch = z0_batch_cpu.to(target_device)
        context_seq = context_seq_cpu.to(target_device)
        context_padding_mask = context_padding_mask_cpu.to(target_device)

        optimizer.zero_grad()
        t = torch.randint(0, num_timesteps, (batch_size,), device=target_device).long()
        zt_batch, noise_batch = noise_latent(z0_batch, t, diff_params, target_device)

        # --- Classifier-Free Guidance for Sequence Context ---
        context_final = context_seq.clone() # Clone to modify
        mask_final = context_padding_mask.clone() # Clone to modify

        for i in range(batch_size):
            if random.random() < cfg_prob:
                # Replace with zero embeddings for unconditional guidance
                context_final[i] = torch.zeros_like(context_final[i])
                if mask_final is not None:
                    # For zero inputs, typically treat all tokens as valid (not padded)
                    mask_final[i] = torch.zeros_like(mask_final[i], dtype=torch.bool)

        # --- Forward pass with sequence context ---
        predicted_noise = model(zt_batch, t, context_final, mask_final) # Output: [B, N, latent_dim]

        loss = criterion(predicted_noise, noise_batch)
        if torch.isnan(loss).any(): print("NaN Loss!"); continue # Skip step
        loss.backward()
        optimizer.step()

        if global_step % log_interval == 0:
            pbar.set_postfix({"Step": global_step, "Loss": f"{loss.item():.4f}"})

        if global_step % eval_interval == 0 and global_step > 0:
            print(f"\n--- Evaluating at step {global_step} ---")
            model.eval()
            total_eval_loss = 0.0; num_eval_batches = 0
            for _ in range(5): # Simulate 5 eval batches
                z0_eval_cpu = torch.randn(batch_size, num_svg_tokens, latent_dim) * 2.0 - 1.0
                sim_text_seq_len_eval = max_text_seq_len
                context_seq_eval_cpu = torch.randn(batch_size, sim_text_seq_len_eval, context_dim)
                context_padding_mask_eval_cpu = torch.zeros(batch_size, sim_text_seq_len_eval, dtype=torch.bool)

                z0_eval = z0_eval_cpu.to(target_device)
                context_seq_eval = context_seq_eval_cpu.to(target_device)
                context_padding_mask_eval = context_padding_mask_eval_cpu.to(target_device)

                t_eval = torch.randint(0, num_timesteps, (batch_size,), device=target_device).long()
                zt_eval, noise_eval = noise_latent(z0_eval, t_eval, diff_params, target_device)
                with torch.no_grad():
                    predicted_noise_eval = model(zt_eval, t_eval, context_seq_eval, context_padding_mask_eval)
                    eval_loss = criterion(predicted_noise_eval, noise_eval)
                if not torch.isnan(eval_loss):
                     total_eval_loss += eval_loss.item(); num_eval_batches += 1

            avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else float('inf')
            print(f"Avg Eval Loss (approx): {avg_eval_loss:.4f}")

            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"✨ New best model saved with Eval Loss: {best_eval_loss:.4f} to {best_model_path} ✨")

            print("Running DDIM sampling simulation...")
            # Simulate context for sampling
            sim_text_seq_len_samp = max_text_seq_len
            num_samples_to_generate = min(4, batch_size)
            # Create a dummy prompt embedding (e.g., for "a cat") and its mask
            dummy_prompt = "a cat"
            dummy_clip_inputs = clip_tokenizer(
                [dummy_prompt] * num_samples_to_generate,
                padding="max_length",
                max_length=sim_text_seq_len_samp,
                truncation=True,
                return_tensors="pt"
            ).to(target_device)
            dummy_clip_outputs = clip_model_sim(**dummy_clip_inputs)
            eval_sample_context_seq = dummy_clip_outputs.last_hidden_state # [B, S_text, D_clip]
            eval_sample_padding_mask = ~(dummy_clip_inputs.attention_mask.bool()) # [B, S_text]
            
            generated_z0, _ = ddim_sample( # Unpack second return item (stats)
                model=model, shape=(num_samples_to_generate, num_svg_tokens, latent_dim), # Pass N
                context_seq=eval_sample_context_seq,
                context_padding_mask=eval_sample_padding_mask,
                diff_params=diff_params,
                num_timesteps=num_timesteps, target_device=target_device,
                cfg_scale=cfg_scale_eval, eta=0.0,
                clip_model=clip_model_sim, clip_tokenizer=clip_tokenizer_sim,
                return_visuals=False # Set to True to see sampling plots
            )
            print(f"Generated z0 shape: {generated_z0.shape}") # Should be [B, N, latent_dim]
            print(f"Generated z0 mean: {generated_z0.mean().item():.4f}, std: {generated_z0.std().item():.4f}")
            model.train()
            print("--- Evaluation complete ---")

        global_step += 1
        pbar.update(1)

    pbar.close()
    print("Training finished!"); print(f"Best eval loss: {best_eval_loss:.4f}")
    if os.path.exists(best_model_path): print(f"Best model saved: {best_model_path}")
