# --- START OF FILE test_vsdit_hidden_seqcond.py --- # Renamed for clarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm  # For progress bars
import os # Needed for saving models
import random

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Helper Modules (Unchanged) ---

def apply_rope(x, seq_dim=1):
    """
    Applies Rotary Positional Embeddings (RoPE) to the input tensor.
    
    Args:
        x (Tensor): Shape [B, L, D], where D must be even.
        seq_dim (int): The sequence dimension. Default: 1 (e.g., [B, L, D])
    
    Returns:
        Tensor: same shape as x, with RoPE applied.
    """
    bsz, seqlen, dim = x.shape
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE."

    # Compute rotary angles
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    pos = torch.arange(seqlen, dtype=torch.float32, device=x.device)
    sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)  # [L, D/2]
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()      # [L, D/2]

    # Expand and reshape
    sin = sin[None, :, :].repeat(bsz, 1, 1)                # [B, L, D/2]
    cos = cos[None, :, :].repeat(bsz, 1, 1)

    x1, x2 = x[..., :half_dim], x[..., half_dim:]          # Split [B, L, D] -> 2x [B, L, D/2]

    # Rotate
    x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rotated


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
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.act2 = act_layer()
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act1(x); x = self.drop(x)
        x = self.fc2(x); x = self.act2(x); x = self.drop(x)
        x = self.fc3(x); x = self.drop(x)
        return x

# --- VS-DiT Block (Modified for Sequence Context) ---

class VS_DiT_Block(nn.Module):
    """ A VS-DiT block operating with internal hidden_dim, accepting sequence context. """
    def __init__(self, hidden_dim, context_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim; self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        modulation_dim = hidden_dim * 9
        self.modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, modulation_dim, bias=True))
        # Context projection projects from context_dim to hidden_dim (applied on last dim of sequence)
        #self.context_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        #nn.init.zeros_(self.context_proj.weight); nn.init.zeros_(self.context_proj.bias)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        # Cross attention kdim/vdim are now hidden_dim (after projection)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout, kdim=hidden_dim, vdim=hidden_dim)
        mlp_internal_dim = int(hidden_dim * mlp_ratio)
        self.mlp = MLP(in_features=hidden_dim, hidden_features=mlp_internal_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=dropout)

    def _compute_modulation(self, t_emb):
        modulation = self.modulation_mlp(t_emb)
        mod = modulation.chunk(9, dim=1)
        return mod

    # --- MODIFIED forward signature ---
    def forward(self, x, t_emb, context_seq, context_padding_mask=None):
        # x: [B, hidden_dim]
        # t_emb: [B, hidden_dim]
        # context_seq: [B, S, context_dim] (e.g., last_hidden_state)
        # context_padding_mask: [B, S] (True where padded)
        B, D_hidden = x.shape
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2, gamma_3, beta_3, alpha_3 = self._compute_modulation(t_emb)

        # Reshape latent input for attention: [B, 1, hidden_dim]
        x_q = x.unsqueeze(1) # Use x_q for query in attentions

        # --- MODIFIED: Project the whole context sequence ---
        # Input context_seq shape: [B, S, context_dim]
        # Output projected_context_seq shape: [B, S, hidden_dim]
        #projected_context_seq = self.context_proj(context_seq)

        

        # --- Self-Attention on latent ---
        normed_x_sa = self.norm1(x_q)
        mod_x_sa = normed_x_sa * (1 + gamma_1.unsqueeze(1)) + beta_1.unsqueeze(1)
        sa_out, _ = self.self_attn(mod_x_sa, mod_x_sa, mod_x_sa, need_weights=False)
        x_q = x_q + alpha_1.unsqueeze(1) * sa_out # Residual update on the query sequence

        # --- Cross-Attention (Query: latent, Key/Value: context sequence) ---
        normed_x_ca = self.norm2(x_q)
        mod_x_ca = normed_x_ca * (1 + gamma_2.unsqueeze(1)) + beta_2.unsqueeze(1)
        # --- MODIFIED: Pass full projected sequence and padding mask ---
        ca_out, _ = self.cross_attn(
            query=mod_x_ca,
            key=context_seq,
            value=context_seq,
            key_padding_mask=context_padding_mask, # Use the mask here
            need_weights=False
        )
        x_q = x_q + alpha_2.unsqueeze(1) * ca_out # Residual update on the query sequence

        # --- FeedForward on latent ---
        normed_x_ff = self.norm3(x_q)
        mod_x_ff = normed_x_ff * (1 + gamma_3.unsqueeze(1)) + beta_3.unsqueeze(1)
        ff_out = self.mlp(mod_x_ff)
        x_q = x_q + alpha_3.unsqueeze(1) * ff_out # Residual update on the query sequence

        # Return dimension [B, hidden_dim]
        output = x_q.squeeze(1)
        return output

# --- VS-DiT Model (Modified for Sequence Context) ---

class VS_DiT(nn.Module):
    """ VS-DiT model accepting sequence context. """
    def __init__(self, latent_dim, hidden_dim, context_dim, num_blocks, num_heads, mlp_ratio=8.0, dropout=0.1): # Example using ratio 8
        super().__init__()
        self.latent_dim = latent_dim; self.hidden_dim = hidden_dim; self.num_blocks = num_blocks
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_dim)
        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.blocks = nn.ModuleList([
            VS_DiT_Block(
                hidden_dim=hidden_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, dropout=dropout
            ) for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim, bias=True)
        #self.initialize_weights()

    def initialize_weights_v0(self):
        nn.init.xavier_uniform_(self.proj_in.weight); nn.init.zeros_(self.proj_in.bias)
        nn.init.zeros_(self.final_proj.weight); nn.init.zeros_(self.final_proj.bias)
        #nn.init.zeros_(self.context_proj.weight); nn.init.zeros_(self.context_proj.bias)
        for block in self.blocks:
            if hasattr(block, 'modulation_mlp') and isinstance(block.modulation_mlp[-1], nn.Linear):
                nn.init.zeros_(block.modulation_mlp[-1].weight); nn.init.zeros_(block.modulation_mlp[-1].bias)
        if isinstance(self.final_modulation_mlp[-1], nn.Linear):
            nn.init.zeros_(self.final_modulation_mlp[-1].weight); nn.init.zeros_(self.final_modulation_mlp[-1].bias)
    
    def initialize_weights_v1(self):
        # Input projection: Xavier/Glorot initialization for better gradient flow
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        
        # Context projection: Xavier/Glorot initialization
        # Important for text conditioning
        #nn.init.xavier_uniform_(self.context_proj.weight)
        #nn.init.zeros_(self.context_proj.bias)
        
        # Final projection: Initialize to zero
        # Common in diffusion models to start with identity mapping
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        
        # Modulation MLPs in blocks: Initialize to small values
        for block in self.blocks:
            if hasattr(block, 'modulation_mlp') and isinstance(block.modulation_mlp[-1], nn.Linear):
                # Small initialization for modulation to start close to identity
                nn.init.zeros_(block.modulation_mlp[-1].weight)
                nn.init.zeros_(block.modulation_mlp[-1].bias)
        
        # Final modulation MLP: Initialize to small values
        if isinstance(self.final_modulation_mlp[-1], nn.Linear):
            nn.init.normal_(self.final_modulation_mlp[-1].weight, std=0.02)
            nn.init.zeros_(self.final_modulation_mlp[-1].bias)
    
    def initialize_weights(self):
            
            # 1. Basic initialization for all linear layers (will be overridden for specific layers)
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # 2. Specific initialization for proj_in (already good, xavier_uniform_ via _basic_init is fine)
            #    If you want to be explicit:
            nn.init.xavier_uniform_(self.proj_in.weight)
            if self.proj_in.bias is not None:
                nn.init.constant_(self.proj_in.bias, 0)

            # 3. Initialize context_proj to zeros (OVERRIDE _basic_init for this layer)
            #    This is CRITICAL for starting with context "off"
            if hasattr(self, 'context_proj'): # If it exists (it does in the seqcond version)
                nn.init.xavier_uniform_(self.context_proj.weight)
                if self.context_proj.bias is not None:
                    nn.init.constant_(self.context_proj.bias, 0)

            # 4. Initialize timestep embedding MLP (OVERRIDE _basic_init)
            if hasattr(self.t_embedder, 'mlp'):
                nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
                if self.t_embedder.mlp[0].bias is not None:
                    nn.init.constant_(self.t_embedder.mlp[0].bias,0) # Or constant_(..., 0)
                nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
                if self.t_embedder.mlp[2].bias is not None:
                    nn.init.constant_(self.t_embedder.mlp[2].bias, 0) # Or constant_(..., 0)

            # 5. Zero-out output of adaLN modulation MLPs in DiT blocks (OVERRIDE _basic_init)
            for block in self.blocks:
                if hasattr(block, 'modulation_mlp') and isinstance(block.modulation_mlp[-1], nn.Linear):
                    nn.init.constant_(block.modulation_mlp[-1].weight, 0)
                    if block.modulation_mlp[-1].bias is not None:
                        nn.init.constant_(block.modulation_mlp[-1].bias, 0)

            # 6. Zero-out output layers: final_modulation_mlp and final_proj (OVERRIDE _basic_init)
            if isinstance(self.final_modulation_mlp[-1], nn.Linear):
                nn.init.constant_(self.final_modulation_mlp[-1].weight, 0)
                if self.final_modulation_mlp[-1].bias is not None:
                    nn.init.constant_(self.final_modulation_mlp[-1].bias, 0)

            nn.init.constant_(self.final_proj.weight, 0)
            if self.final_proj.bias is not None:
                nn.init.constant_(self.final_proj.bias, 0)

    def initialize_weights_v3(self):
        # 1. Basic initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2. Input projection (good for shape learning)
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)

        # 3. Context projection (balanced for color conditioning)
        if hasattr(self, 'context_proj'):
            # Small but non-zero initialization for context
            nn.init.normal_(self.context_proj.weight, std=0.01)
            nn.init.zeros_(self.context_proj.bias)

        # 4. Modulation networks (for shape preservation)
        for block in self.blocks:
            if hasattr(block, 'modulation_mlp') and isinstance(block.modulation_mlp[-1], nn.Linear):
                # Small initialization for stable shape learning
                nn.init.normal_(block.modulation_mlp[-1].weight, std=0.001)
                nn.init.zeros_(block.modulation_mlp[-1].bias)

        # 5. Final layers
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        
        if isinstance(self.final_modulation_mlp[-1], nn.Linear):
            nn.init.normal_(self.final_modulation_mlp[-1].weight, std=0.001)
            nn.init.zeros_(self.final_modulation_mlp[-1].bias)

        # 6. Timestep embedding (balanced initialization)
        if hasattr(self.t_embedder, 'mlp'):
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.01)
            nn.init.zeros_(self.t_embedder.mlp[0].bias)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.01)
            nn.init.zeros_(self.t_embedder.mlp[2].bias)
    # --- MODIFIED forward signature ---
    def forward(self, z_t, t, context_seq, context_padding_mask=None):
        # z_t: [B, latent_dim]
        # t: [B]
        # context_seq: [B, S, context_dim]
        # context_padding_mask: [B, S]
        model_device = next(self.parameters()).device
        z_t = z_t.to(model_device); t = t.to(model_device)
        context_seq = context_seq.to(model_device)
        if context_padding_mask is not None:
            context_padding_mask = context_padding_mask.to(model_device)

        t_emb = self.t_embedder(t) # Output: [B, hidden_dim]
        h = self.proj_in(z_t)      # Output: [B, hidden_dim]

        # --- MODIFIED: Project context sequence to hidden_dim ---
        proj_context_seq = self.context_proj(context_seq) # Output: [B, S, hidden_dim]

        # --- MODIFIED: Pass sequence context and mask to blocks ---
        for block in self.blocks:
            h = block(h, t_emb, proj_context_seq, context_padding_mask) # Output: [B, hidden_dim]

        final_gamma, final_beta = self.final_modulation_mlp(t_emb).chunk(2, dim=1)
        normed_h = self.final_norm(h)
        mod_h = normed_h * (1 + final_gamma) + final_beta
        epsilon_pred = self.final_proj(mod_h) # Output: [B, latent_dim]
        return epsilon_pred

# --- Diffusion Utilities (Unchanged) ---

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
    z0 = z0.to(target_device); t = t.to(target_device)
    sqrt_alpha_bar = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1)
    sqrt_one_minus_alpha_bar = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1)
    epsilon = torch.randn_like(z0, device=target_device)
    zt = sqrt_alpha_bar * z0 + sqrt_one_minus_alpha_bar * epsilon
    return zt, epsilon

# --- Sampling (DDIM) (Modified for Sequence Context) ---

@torch.no_grad()
def ddim_sample_v0(model, shape, context_seq, context_padding_mask, # Changed context args
                diff_params, num_timesteps, target_device, cfg_scale=3.0, eta=0.0):
    """ DDIM sampling process accepting sequence context and mask. """
    model.eval(); batch_size = shape[0]
    z_t = torch.randn(shape, device=target_device)

    # Ensure conditional inputs are on the correct device
    context_seq = context_seq.to(target_device)
    if context_padding_mask is not None:
        context_padding_mask = context_padding_mask.to(target_device)

    # --- MODIFIED: Handle unconditional input for sequences ---
    # Create zero tensors for unconditional sequence. This is a simplification.
    # Using embeddings of "" is often preferred.
    uncond_context_seq = torch.zeros_like(context_seq)
    # For unconditional, assume no padding (all tokens are valid for the zero input)
    uncond_padding_mask = torch.zeros_like(context_padding_mask, dtype=torch.bool) if context_padding_mask is not None else None

    for i in tqdm(reversed(range(num_timesteps)), desc="DDIM Sampling", total=num_timesteps, leave=False):
        t = torch.full((batch_size,), i, dtype=torch.long, device=target_device)
        alpha_bar_t = diff_params["alphas_cumprod"][t].view(-1, 1)
        alpha_bar_t_prev = diff_params["alphas_cumprod_prev"][t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1)
        sqrt_alpha_bar_t = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1)
        beta_t = diff_params["betas"][t].view(-1, 1)

        # --- MODIFIED: Call model with sequence context and mask ---
        eps_cond = model(z_t, t, context_seq, context_padding_mask)
        eps_uncond = model(z_t, t, uncond_context_seq, uncond_padding_mask) # Use uncond inputs
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        #eps_pred = eps_cond

        pred_z0 = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        pred_z0 = torch.clamp(pred_z0, -5, 5) # Clamp based on typical latent range

        variance = (1. - alpha_bar_t_prev) / (1. - alpha_bar_t) * beta_t
        variance = torch.clamp(variance, min=1e-20)
        sigma_t = eta * torch.sqrt(variance)
        pred_dir_t = torch.sqrt(torch.clamp(1. - alpha_bar_t_prev - sigma_t**2, min=0.0)) * eps_pred
        z_t_prev = torch.sqrt(alpha_bar_t_prev) * pred_z0 + pred_dir_t
        if eta > 0:
            noise = torch.randn_like(z_t, device=target_device)
            z_t_prev += sigma_t * noise
        z_t = z_t_prev

    #model.train()
    return z_t

@torch.no_grad()
def ddim_sample_v1(model, shape, context_seq, context_padding_mask,
                diff_params, num_timesteps, target_device, cfg_scale=3.0, eta=0.0, clip_model=None, clip_tokenizer=None):
    """DDIM sampling with correct formulation."""
    model.eval()
    batch_size = shape[0]
    z_t = torch.randn(shape, device=target_device)

    # Move context to device
    context_seq = context_seq.to(target_device)
    if context_padding_mask is not None:
        context_padding_mask = context_padding_mask.to(target_device)

    # Setup unconditional context
    #uncond_context_seq = torch.zeros_like(context_seq)
    #uncond_padding_mask = torch.zeros_like(context_padding_mask, dtype=torch.bool) if context_padding_mask is not None else None

    # Get empty string embedding instead of zeros
    empty_text = clip_tokenizer("", padding='max_length', truncation=True, return_tensors="pt").to(device)
    empty_outputs = clip_model(**empty_text)
    uncond_context_seq = empty_outputs.last_hidden_state.repeat(batch_size, 1, 1)
    uncond_padding_mask = ~(empty_text.attention_mask.bool()).repeat(batch_size, 1)

    for i in tqdm(reversed(range(num_timesteps)), desc="DDIM Sampling"):
        t = torch.full((batch_size,), i, dtype=torch.long, device=target_device)
        alpha_bar_t = diff_params["alphas_cumprod"][t].view(-1, 1)
        alpha_bar_t_prev = diff_params["alphas_cumprod_prev"][t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1)
        sqrt_alpha_bar_t = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1)

        # Get noise predictions
        eps_cond = model(z_t, t, context_seq, context_padding_mask)
        eps_uncond = model(z_t, t, uncond_context_seq, uncond_padding_mask)
        eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        #eps_pred = eps_cond  # Uncomment to use only conditional

        # Compute true DDIM sigma_t
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * 
            (1 - alpha_bar_t / alpha_bar_t_prev)
        )

        # Predict x₀
        pred_x0 = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        pred_x0 = torch.clamp(pred_x0, -5, 5)

        # Deterministic part
        dir_xt = torch.sqrt(alpha_bar_t_prev) * pred_x0
        noise_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * eps_pred

        # Combine deterministic and stochastic parts
        if eta > 0:
            z_t_prev = dir_xt + noise_xt + sigma_t * torch.randn_like(z_t)
        else:
            z_t_prev = dir_xt + noise_xt

        if i % 100 == 0:  # Debug prints
            print(f"\nStep {i} stats:")
            print(f"eps_cond mean: {eps_cond.mean():.4f}, std: {eps_cond.std():.4f}")
            print(f"eps_uncond mean: {eps_uncond.mean():.4f}, std: {eps_uncond.std():.4f}")
            print(f"pred_x0 mean: {pred_x0.mean():.4f}, std: {pred_x0.std():.4f}")
            print(f"z_t_prev mean: {z_t_prev.mean():.4f}, std: {z_t_prev.std():.4f}")

        z_t = z_t_prev

    return z_t

@torch.no_grad()
def ddim_sample(model, shape, context_seq, context_padding_mask,
                diff_params, num_timesteps, target_device, cfg_scale=3.0, eta=0.0, clip_model=None, clip_tokenizer=None):
    """DDIM sampling with correct formulation."""
    model.eval()
    batch_size = shape[0]
    z_t = torch.randn(shape, device=target_device)

    # Debug initial noise
    print("\nInitial noise stats:")
    print(f"z_t mean: {z_t.mean():.4f}, std: {z_t.std():.4f}")

    # Move context to device
    context_seq = context_seq.to(target_device)
    if context_padding_mask is not None:
        context_padding_mask = context_padding_mask.to(target_device)

    # Get empty string embedding with proper padding
    max_length = context_seq.size(1)  # Use conditional length
    empty_text = clip_tokenizer(
        "" * batch_size,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    ).to(target_device)
    
    # Get unconditional embeddings
    empty_outputs = clip_model(**empty_text)
    uncond_context_seq = empty_outputs.last_hidden_state
    
    # Debug embeddings
    print("\nEmbedding shapes before repeat:")
    print(f"Conditional: {context_seq.shape}")
    print(f"Unconditional: {uncond_context_seq.shape}")
    
    # Repeat for batch size
    uncond_context_seq = uncond_context_seq.repeat(batch_size, 1, 1)
    uncond_padding_mask = ~(empty_text.attention_mask.bool()).repeat(batch_size, 1)

    print("\nFinal shapes:")
    print(f"Conditional: {context_seq.shape}")
    print(f"Unconditional: {uncond_context_seq.shape}")
    print(f"Conditional mask: {context_padding_mask.shape}")
    print(f"Unconditional mask: {uncond_padding_mask.shape}")

    for i in tqdm(reversed(range(num_timesteps)), desc="DDIM Sampling"):
        t = torch.full((batch_size,), i, dtype=torch.long, device=target_device)
        alpha_bar_t = diff_params["alphas_cumprod"][t].view(-1, 1)
        alpha_bar_t_prev = diff_params["alphas_cumprod_prev"][t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = diff_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1)
        sqrt_alpha_bar_t = diff_params["sqrt_alphas_cumprod"][t].view(-1, 1)

        # Get noise predictions
        eps_cond = model(z_t, t, context_seq, context_padding_mask)
        eps_uncond = model(z_t, t, uncond_context_seq, uncond_padding_mask)
        
        # Debug predictions if they're identical
        if i % 100 == 0:
            diff = (eps_cond - eps_uncond).abs().mean().item()
            print(f"\nStep {i} prediction difference: {diff:.6f}")

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(eps_cond.flatten(1), eps_uncond.flatten(1), dim=1).mean().item()

        # # Dynamic CFG scale
        # if cosine_sim > 0.99:
        #     current_cfg = cfg_scale * 2.0
        # elif cosine_sim < 0.99:
        #     current_cfg = cfg_scale * 0.7
        # else:
        #     current_cfg = cfg_scale

        # Linear scaling between [0.90, 0.99]
        min_sim = 0.90
        max_sim = 0.99

        # Clamp cosine sim
        cosine_sim_clamped = max(min(cosine_sim, max_sim), min_sim)
        scale_factor = 1 + ((cosine_sim_clamped - min_sim) / (max_sim - min_sim))  # from 1.0 to 2.0
        current_cfg = cfg_scale * scale_factor

        #current_cfg = max(current_cfg, cfg_scale * 0.5)
        
        eps_pred = eps_uncond + current_cfg * (eps_cond - eps_uncond)
        #eps_pred = (1 - cfg_scale) * eps_uncond + cfg_scale * eps_cond
        #eps_pred = (1 + cfg_scale) * eps_cond - cfg_scale * eps_uncond

        # Compute true DDIM sigma_t
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * 
            (1 - alpha_bar_t / alpha_bar_t_prev)
        )

        # Predict x₀
        pred_x0 = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
        #print("before clamping")
        #print(pred_x0.min(), pred_x0.max())
        pred_x0 = torch.clamp(pred_x0, -5, 5)

        # Deterministic part
        dir_xt = torch.sqrt(alpha_bar_t_prev) * pred_x0
        noise_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * eps_pred

        # Combine deterministic and stochastic parts
        if eta > 0:
            z_t_prev = dir_xt + noise_xt + sigma_t * torch.randn_like(z_t)
        else:
            z_t_prev = dir_xt + noise_xt

        if i % 100 == 0:  # Debug prints
            cosine_sim = F.cosine_similarity(eps_cond.flatten(1), eps_uncond.flatten(1), dim=1).mean().item()
            print(f"\nStep {i} stats:")
            print(f"eps_cond mean: {eps_cond.mean():.4f}, std: {eps_cond.std():.4f}")
            print(f"eps_uncond mean: {eps_uncond.mean():.4f}, std: {eps_uncond.std():.4f}")
            print(f"pred_x0 mean: {pred_x0.mean():.4f}, std: {pred_x0.std():.4f}")
            print(f"z_t_prev mean: {z_t_prev.mean():.4f}, std: {z_t_prev.std():.4f}")
            print(f"eps cosine similarity: {cosine_sim:.4f}")

        z_t = z_t_prev

    print("\nFinal output stats:")
    print(f"z_t mean: {z_t.mean():.4f}, std: {z_t.std():.4f}")
    
    return z_t



# --- Training (Modified for Sequence Context Simulation/Eval) ---

def train(model, train_steps, batch_size, latent_dim, context_dim, # context_dim now refers to embed dim per token
          num_timesteps, cfg_prob, log_interval, eval_interval, cfg_scale_eval, target_device,
          model_save_dir="saved_models", max_seq_len=77): # Added max_seq_len for simulation

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate']) # Use config LR
    criterion = nn.MSELoss()
    betas_cpu = get_linear_noise_schedule(num_timesteps)
    diff_params = precompute_diffusion_parameters(betas_cpu, target_device)
    model.to(target_device); model.train()
    best_eval_loss = float('inf')
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "vs_dit_seqcond_best.pth") # New name
    global_step = 0
    pbar = tqdm(total=train_steps, desc="Training")

    while global_step < train_steps:
        # --- Simulate a batch of data (on CPU initially) ---
        z0_batch_cpu = torch.randn(batch_size, latent_dim) * 2.0 - 1.0
        # --- MODIFIED: Simulate sequence context and mask ---
        sim_seq_len = random.randint(10, max_seq_len) # Simulate variable length
        context_seq_cpu = torch.randn(batch_size, sim_seq_len, context_dim)
        # Create padding mask (True where padded)
        context_padding_mask_cpu = torch.zeros(batch_size, sim_seq_len, dtype=torch.bool) # All False initially
        # If simulating padding, adjust mask here. For simplicity, assume no padding in sim.

        z0_batch = z0_batch_cpu.to(target_device)
        context_seq = context_seq_cpu.to(target_device)
        context_padding_mask = context_padding_mask_cpu.to(target_device)

        optimizer.zero_grad()
        t = torch.randint(0, num_timesteps, (batch_size,), device=target_device).long()
        zt_batch, noise_batch = noise_latent(z0_batch, t, diff_params, target_device)

        # --- MODIFIED: Classifier-Free Guidance for Sequence ---
        # Randomly replace sequence/mask with zeros/valid mask for unconditional
        context_final = context_seq
        mask_final = context_padding_mask
        for i in range(batch_size):
            if random.random() < cfg_prob:
                context_final[i] = torch.zeros_like(context_final[i])
                if mask_final is not None:
                    mask_final[i] = torch.zeros_like(mask_final[i], dtype=torch.bool) # All valid for zero input

        # --- MODIFIED: Forward pass with sequence context ---
        predicted_noise = model(zt_batch, t, context_final, mask_final)

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
                z0_eval_cpu = torch.randn(batch_size, latent_dim) * 2.0 - 1.0
                # Simulate eval sequence context/mask
                sim_seq_len_eval = random.randint(10, max_seq_len)
                context_seq_eval_cpu = torch.randn(batch_size, sim_seq_len_eval, context_dim)
                context_padding_mask_eval_cpu = torch.zeros(batch_size, sim_seq_len_eval, dtype=torch.bool)

                z0_eval = z0_eval_cpu.to(target_device)
                context_seq_eval = context_seq_eval_cpu.to(target_device)
                context_padding_mask_eval = context_padding_mask_eval_cpu.to(target_device)

                t_eval = torch.randint(0, num_timesteps, (batch_size,), device=target_device).long()
                zt_eval, noise_eval = noise_latent(z0_eval, t_eval, diff_params, target_device)
                with torch.no_grad():
                    # Evaluate with conditional context
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

            print("Running DDIM sampling...")
            # Simulate context for sampling
            sim_seq_len_samp = random.randint(10, max_seq_len)
            num_samples = min(4, batch_size)
            eval_sample_context_seq = torch.randn(num_samples, sim_seq_len_samp, context_dim, device=target_device)
            eval_sample_padding_mask = torch.zeros(num_samples, sim_seq_len_samp, dtype=torch.bool, device=target_device)

            # --- MODIFIED: Call ddim_sample with sequence context ---
            generated_z0 = ddim_sample(
                model=model, shape=(num_samples, latent_dim),
                context_seq=eval_sample_context_seq,
                context_padding_mask=eval_sample_padding_mask,
                diff_params=diff_params,
                num_timesteps=num_timesteps, target_device=target_device,
                cfg_scale=cfg_scale_eval, eta=0.0
            )
            print(f"Generated z0 shape: {generated_z0.shape}")
            print(f"Generated z0 mean: {generated_z0.mean().item():.4f}, std: {generated_z0.std().item():.4f}")
            model.train()
            print("--- Evaluation complete ---")

        global_step += 1
        pbar.update(1)

    pbar.close()
    print("Training finished!"); print(f"Best eval loss: {best_eval_loss:.4f}")
    if os.path.exists(best_model_path): print(f"Best model saved: {best_model_path}")


# --- Main Execution (Example - Adapt for your actual training) ---
if __name__ == '__main__':
    # --- Model & Training Parameters ---
    LATENT_DIM = 128
    HIDDEN_DIM = 384
    CONTEXT_DIM = 768 # Dimension of CLIP last_hidden_state (e.g., 768 for ViT-B/32)
    NUM_BLOCKS = 12
    NUM_HEADS = 6
    MLP_RATIO = 8.0 # Example deeper ratio
    DROPOUT = 0.1
    MAX_SEQ_LEN = 77 # Max sequence length from CLIP tokenizer

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    TRAIN_STEPS = 1001 # Keep short for testing this script
    NUM_TIMESTEPS = 1000

    CFG_PROB = 0.1
    CFG_SCALE_EVAL = 4.0
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 250
    MODEL_SAVE_DIR = "saved_models_vsdit_seqcond" # New directory

    if HIDDEN_DIM % NUM_HEADS != 0: raise ValueError("HIDDEN_DIM must be divisible by NUM_HEADS")

    config = locals() # Capture parameters in a dict for potential logging/use

    vs_dit_model = VS_DiT(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, context_dim=CONTEXT_DIM,
        num_blocks=NUM_BLOCKS, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT
    )
    print(f"Model Parameters: {sum(p.numel() for p in vs_dit_model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # NOTE: This runs training with SIMULATED sequence data.
    # You would replace the train function call with your actual
    # training loop from train_vsdit_with_clip.py, passing real data.
    print("\n--- Running internal training simulation ---")
    train(
        model=vs_dit_model, train_steps=TRAIN_STEPS, batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM, context_dim=CONTEXT_DIM, learning_rate=LEARNING_RATE,
        num_timesteps=NUM_TIMESTEPS, cfg_prob=CFG_PROB, log_interval=LOG_INTERVAL,
        eval_interval=EVAL_INTERVAL, cfg_scale_eval=CFG_SCALE_EVAL, target_device=device,
        model_save_dir=MODEL_SAVE_DIR, max_seq_len=MAX_SEQ_LEN
    )

# --- END OF FILE test_vsdit_hidden_seqcond.py ---