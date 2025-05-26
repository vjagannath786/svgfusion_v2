# --- START OF FILE vpvae_accelerate_ce.py ---
import sys
sys.path.append('/home/svgfusion_v2/') # Or your project root

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import wandb
import math
import random
import os
import traceback

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # For debugging CUDA errors

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed as accelerate_set_seed

from svgutils import DynamicProgressiveSVGDataset # Assumes this will provide data in new format
from svgutils import SVGToTensor_Normalized    # Used for token defs & vocab sizes

# --- Set Seed Utility ---
def set_seed(seed, accelerator=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if accelerator is not None: accelerate_set_seed(seed)
    else:
        if torch.backends.mps.is_available():
            try: torch.mps.manual_seed(seed)
            except AttributeError: pass

# --- RoPE Utility ---
def apply_rope(x, seq_dim=1):
    if x.shape[-1] == 0: return x
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

# --- Model Definitions ---
class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, num_heads, kdim=None, vdim=None, batch_first=True):
        super().__init__(); self.num_heads=num_heads; self.d_model=d_model; assert d_model % num_heads == 0; self.batch_first=batch_first
        self.kdim = kdim if kdim is not None else d_model; self.vdim = vdim if vdim is not None else d_model; self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model); self.w_k = nn.Linear(self.kdim, d_model); self.w_v = nn.Linear(self.vdim, d_model); self.w_o = nn.Linear(d_model, d_model)
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size=query.size(0); q=self.w_q(query); k=self.w_k(key); v=self.w_v(value); q=q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2); k=k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2); v=v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores=torch.matmul(q, k.transpose(-2, -1))/math.sqrt(self.d_k);
        if key_padding_mask is not None: scores=scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        if attn_mask is not None:
             if attn_mask.dim()==2: attn_mask=attn_mask.unsqueeze(0).unsqueeze(0)
             scores=scores.masked_fill(attn_mask, float('-inf'))
        attn=F.softmax(scores, dim=-1); context=torch.matmul(attn, v); context=context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model); output=self.w_o(context); return output

class TransformerBlock(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff=512, dropout=0.1, batch_first=True): super().__init__(); self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=batch_first); self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model); self.feed_forward=nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)); self.dropout=nn.Dropout(dropout)
    def forward(self, x, padding_mask=None, attn_mask=None): attn_output,_=self.self_attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask, need_weights=False); x=self.norm1(x+self.dropout(attn_output)); ff_output=self.feed_forward(x); x=self.norm2(x+self.dropout(ff_output)); return x

class VPVAEEncoder(nn.Module):
    def __init__(self, num_element_types, num_command_types, 
                 element_embed_dim, command_embed_dim,
                 num_other_continuous_svg_features, # e.g., 8 (geo) + 4 (style) = 12
                 pixel_feature_dim, d_model, num_layers, num_heads, latent_dim, max_seq_len,
                 element_padding_idx=0, command_padding_idx=0, num_bins=256,svg_param_embed_dim=64 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.element_type_embedding = nn.Embedding(num_element_types, element_embed_dim, padding_idx=element_padding_idx)
        self.command_type_embedding = nn.Embedding(num_command_types, command_embed_dim, padding_idx=command_padding_idx)
        
        self.shared_param_embedding = nn.Embedding(num_bins, svg_param_embed_dim)
        self.num_parameters = num_other_continuous_svg_features
        

        self.svg_input_feature_dim = element_embed_dim + command_embed_dim + num_other_continuous_svg_features * svg_param_embed_dim

        self.svg_projection = nn.Linear(self.svg_input_feature_dim, d_model)
        self.pixel_projection = nn.Linear(pixel_feature_dim, d_model)
        #self.svg_norm_pre_proj = nn.LayerNorm(self.svg_input_feature_dim)
        #self.pixel_norm_pre_proj = nn.LayerNorm(pixel_feature_dim)

        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, kdim=d_model, vdim=d_model, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.self_attention_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_model*4, batch_first=True) for _ in range(num_layers)])
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)

    def forward(self, svg_matrix_hybrid, pixel_embedding, svg_padding_mask=None, pixel_padding_mask=None):
        # svg_matrix_hybrid: [B, L_svg, 1(elem_id) + 1(cmd_id) + N_other_cont_params]
        # Example: Total 14 columns -> 1 elem_id, 1 cmd_id, 12 other_cont_params
        
        element_ids = svg_matrix_hybrid[:, :, 0].long()    # [B, L_svg]
        command_ids = svg_matrix_hybrid[:, :, 1].long()    # [B, L_svg]
        other_continuous_params = svg_matrix_hybrid[:, :, 2:].long() # [B, L_svg, N_other_cont_params]

        elem_embeds = self.element_type_embedding(element_ids)     # [B, L_svg, E_elem]
        cmd_embeds = self.command_type_embedding(command_ids)       # [B, L_svg, E_cmd]
        
        
        param_embeds_list = []
        for i in range(self.num_parameters):
             try:
                 # Get indices for the i-th parameter: [B, L_svg]
                 p_indices_i = other_continuous_params[:, :, i]
                 # Pass through the *shared* embedding layer
                 p_embed = self.shared_param_embedding(p_indices_i) # [B, L_svg, E_param]
                 param_embeds_list.append(p_embed)
             except IndexError as e: print(f"Error shared param {i} embed: max {p_indices_i.max()}, num_emb {self.shared_param_embedding.num_embeddings}"); raise e
        
        svg_features_concatenated = torch.cat([elem_embeds, cmd_embeds] + param_embeds_list, dim=-1)
        
        svg_features_rope = apply_rope(svg_features_concatenated)
        #svg_features_normed = self.svg_norm_pre_proj(svg_features_rope)
        svg_projected = self.svg_projection(svg_features_rope) 

        #pixel_features_normed = self.pixel_norm_pre_proj(pixel_embedding) 
        pixel_projected = self.pixel_projection(pixel_embedding)

        cross_attn_output = self.cross_attention(query=svg_projected, key=pixel_projected, value=pixel_projected, key_padding_mask=pixel_padding_mask)
        x = self.cross_attn_norm(svg_projected + cross_attn_output)
        if svg_padding_mask is not None: x = x * (~svg_padding_mask).unsqueeze(-1).float()
        for layer in self.self_attention_layers: x = layer(x, padding_mask=svg_padding_mask)
        # if svg_padding_mask is not None:
        #     valid_mask = (~svg_padding_mask).unsqueeze(-1).float()
        #     pooled_x = (x * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-9)
        # else: pooled_x = torch.mean(x, dim=1)
        # mu = self.fc_mu(pooled_x); log_var = self.fc_var(pooled_x)
        x = x.transpose(1, 2)  # [B, d_model, 1024]
        x = F.avg_pool1d(x, kernel_size=4, stride=4)  # [B, d_model, 256]
        x = x.transpose(1, 2)  # [B, 256, d_model]
        
        
        
        mu = self.fc_mu(x); 
        log_var = self.fc_var(x)
        
        if svg_padding_mask is not None:
            valid_mask = (~svg_padding_mask).float()  # [B, 1024]
            valid_mask = valid_mask.unsqueeze(1)      # [B, 1, 1024]
            valid_mask = F.avg_pool1d(valid_mask, kernel_size=4, stride=4)  # [B, 1, 256]
            valid_mask = (valid_mask > 0.5).float()   # Binarize if needed
            valid_mask = valid_mask.transpose(1, 2)   # [B, 256, 1]
            #valid_mask = (~svg_padding_mask).unsqueeze(-1).float()  # [B, seq_len, 1]
            mu = mu * valid_mask
            log_var = log_var * valid_mask
        return mu, log_var

class VPVAEDecoder(nn.Module):
    def __init__(self, num_element_types, num_command_types, num_other_continuous_params_to_reconstruct, 
                 latent_dim, d_model, num_layers, num_heads, max_seq_len, num_bins=256):
        super().__init__()
        self.max_seq_len = max_seq_len; self.d_model = d_model
        self.num_bins = num_bins
        
        self.fc_latent = nn.Linear(latent_dim, d_model)
        self.decoder_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff=d_model*4, batch_first=True) for _ in range(num_layers)])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        self.element_type_head = nn.Linear(d_model, num_element_types) 
        self.command_type_head = nn.Linear(d_model, num_command_types) 
        # self.other_continuous_params_head = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2), nn.ReLU(),
        #     nn.Linear(d_model // 2, num_other_continuous_params_to_reconstruct),
        #     nn.Tanh() 
        # )
        self.param_heads = nn.ModuleList([
            nn.Linear(d_model, num_bins) for _ in range(num_other_continuous_params_to_reconstruct)
        ])

    def forward(self, z, target_len):
        batch_size=z.size(0); x=self.fc_latent(z); 
        #x=x.repeat(1, target_len, 1)
        #effective_target_len = min(target_len, self.max_seq_len); 
        x_upsampled = F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        x = x_upsampled
        effective_target_len = min(target_len, self.max_seq_len);
        x_rope = apply_rope(x[:,:effective_target_len,:])
        causal_mask=torch.triu(torch.ones(effective_target_len, effective_target_len, device=z.device), diagonal=1).bool()
        for layer in self.decoder_layers: x_rope=layer(x_rope, padding_mask=None, attn_mask=causal_mask)
        x_normed=self.decoder_norm(x_rope)
        
        element_type_logits = self.element_type_head(x_normed)
        command_type_logits = self.command_type_head(x_normed)
        #predicted_other_continuous_params = self.other_continuous_params_head(x_normed)
        param_logits_list = [head(x_normed) for head in self.param_heads]
        
        return element_type_logits, command_type_logits, param_logits_list

class VPVAE(nn.Module):
    def __init__(self, num_element_types, num_command_types, element_embed_dim, command_embed_dim,
                 num_other_continuous_svg_features, # For encoder input
                 num_other_continuous_params_to_reconstruct, # For decoder output
                 pixel_feature_dim, encoder_d_model, decoder_d_model,
                 encoder_layers, decoder_layers, num_heads, latent_dim, max_seq_len,
                 element_padding_idx=0, command_padding_idx=0):
        super().__init__()
        self.encoder=VPVAEEncoder(
            num_element_types, num_command_types, element_embed_dim, command_embed_dim,
            num_other_continuous_svg_features, pixel_feature_dim, encoder_d_model,
            encoder_layers, num_heads, latent_dim, max_seq_len,
            element_padding_idx, command_padding_idx
        )
        self.decoder=VPVAEDecoder(
            num_element_types, num_command_types, num_other_continuous_params_to_reconstruct, 
            latent_dim, decoder_d_model, decoder_layers, num_heads, max_seq_len
        )
        self.max_seq_len=max_seq_len

    def reparameterize(self, mu, logvar): std=torch.exp(0.5*torch.clamp(logvar,min=-10,max=10)); eps=torch.randn_like(std); return mu+eps*std
    
    def forward(self, svg_matrix_hybrid, pixel_embedding, svg_padding_mask=None, pixel_padding_mask=None):
        mu,logvar=self.encoder(svg_matrix_hybrid, pixel_embedding, svg_padding_mask, pixel_padding_mask)
        z=self.reparameterize(mu, logvar)
        element_logits, command_logits, continuous_params_pred = self.decoder(z, target_len=self.max_seq_len)
        return element_logits, command_logits, continuous_params_pred, mu, logvar

def vp_vae_hybrid_loss(
    element_logits, command_logits, param_logits_list,
    target_svg_matrix_hybrid, # Expected: [B, L, 1(elem_id) + 1(cmd_id) + N_other_cont_params]
    mu, logvar,
    element_pad_idx=0, command_pad_idx=0, 
    svg_padding_mask=None, kl_weight=0.1, 
    ce_elem_loss_weight=1.0, ce_cmd_loss_weight=1.0, mse_cont_loss_weight=1.0, num_bins=256,num_params = 12
):
    batch_size, target_len_out, _ = element_logits.shape 
    _, target_len_svg, _ = target_svg_matrix_hybrid.shape
    effective_len = min(target_len_out, target_len_svg)

    target_element_ids = target_svg_matrix_hybrid[:, :effective_len, 0].long()
    target_command_ids = target_svg_matrix_hybrid[:, :effective_len, 1].long()
    target_other_continuous_params = target_svg_matrix_hybrid[:, :effective_len, 2:].long()
    
    element_logits_eff = element_logits[:, :effective_len, :]
    command_logits_eff = command_logits[:, :effective_len, :]
    #predicted_other_continuous_params_eff = param_logits_list[:, :effective_len, :]

    valid_mask_flat = None; mask_sum = float(batch_size * effective_len)
    if svg_padding_mask is not None:
        svg_padding_mask_eff = svg_padding_mask[:, :effective_len]
        valid_mask_flat = (~svg_padding_mask_eff).reshape(-1).float()
        mask_sum = valid_mask_flat.sum() + 1e-9

    ce_loss_elem_unreduced = F.cross_entropy(element_logits_eff.reshape(-1, element_logits_eff.size(-1)), 
                                           target_element_ids.reshape(-1), 
                                           ignore_index=element_pad_idx, reduction='none')
    ce_loss_elem = (ce_loss_elem_unreduced * valid_mask_flat).sum() / mask_sum if valid_mask_flat is not None else ce_loss_elem_unreduced.mean()

    ce_loss_cmd_unreduced = F.cross_entropy(command_logits_eff.reshape(-1, command_logits_eff.size(-1)),
                                          target_command_ids.reshape(-1),
                                          ignore_index=command_pad_idx, reduction='none')
    ce_loss_cmd = (ce_loss_cmd_unreduced * valid_mask_flat).sum() / mask_sum if valid_mask_flat is not None else ce_loss_cmd_unreduced.mean()
    
    # if svg_padding_mask is not None:
    #     cont_feature_valid_mask = (~svg_padding_mask_eff).unsqueeze(-1).expand_as(target_other_continuous_params).float()
    #     num_valid_cont_elements = cont_feature_valid_mask.sum() + 1e-9
    # else:
    #     cont_feature_valid_mask = torch.ones_like(target_other_continuous_params)
    #     num_valid_cont_elements = target_other_continuous_params.numel()

    # mse_loss_cont_unreduced = F.mse_loss(predicted_other_continuous_params_eff * cont_feature_valid_mask,
    #                                      target_other_continuous_params * cont_feature_valid_mask,
    #                                      reduction='none')
    # mse_loss_cont = mse_loss_cont_unreduced.sum() / num_valid_cont_elements if num_valid_cont_elements > 0 else torch.tensor(0.0, device=mu.device)

    # --- 2. Parameter Reconstruction Loss (Sum of CE losses) ---
    total_param_ce_loss = torch.tensor(0.0).to(command_logits.device)
    num_valid_param_losses = 0
    for i in range(num_params):
        if i >= len(param_logits_list): # Safety check
             print(f"Warning: Mismatch between num_params {num_params} and len(param_logits_list) {len(param_logits_list)}")
             continue

        param_logits_i = param_logits_list[i][:, :effective_len, :] # Slice param logits [B, L, num_bins]
        target_indices_i = target_other_continuous_params[:, :, i] # [B, L]

        logits_flat = param_logits_i.reshape(-1, num_bins)
        targets_flat = target_indices_i.reshape(-1)

        # Check if targets are within valid range [0, num_bins-1]
        if targets_flat.min() < 0 or targets_flat.max() >= num_bins:
             print(f"Warning: Target indices for param {i} out of range [0, {num_bins-1}]. Min: {targets_flat.min()}, Max: {targets_flat.max()}. Skipping loss.")
             continue # Skip loss calculation for this parameter if targets are invalid

        ce_loss_p_i_unreduced = F.cross_entropy(logits_flat, targets_flat, reduction='none') # No weights/ignore_index for params assumed

        if svg_padding_mask is not None:
            safe_mask_sum = mask_sum if mask_sum > 1e-6 else 1.0
            ce_loss_p_i = (ce_loss_p_i_unreduced * valid_mask_flat).sum() / safe_mask_sum
        else:
            ce_loss_p_i = ce_loss_p_i_unreduced.mean()

        total_param_ce_loss += ce_loss_p_i
        num_valid_param_losses += 1

    # Average parameter CE loss
    avg_param_ce_loss = total_param_ce_loss / num_valid_param_losses if num_valid_param_losses > 0 else torch.tensor(0.0).to(command_logits.device)
    
    logvar_exp_clamped = torch.clamp(logvar, max=80.0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar_exp_clamped.exp(), dim=1)
    kl_loss = torch.mean(kl_div)

    total_loss = (ce_elem_loss_weight * ce_loss_elem +
                  ce_cmd_loss_weight * ce_loss_cmd +
                  mse_cont_loss_weight * avg_param_ce_loss +
                  kl_weight * kl_loss)
    
    return total_loss, ce_loss_elem, ce_loss_cmd, avg_param_ce_loss, kl_loss

def get_kl_weight(step, total_steps, max_kl_weight=0.1, anneal_portion=0.8, schedule='linear'):
    anneal_steps = int(total_steps * anneal_portion)
    if step < anneal_steps: return max_kl_weight * (step / max(1, anneal_steps))
    else: return max_kl_weight

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, initial_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))  # Warmup
        # Ensure progress never exceeds 1.0 and spans full decay phase
        progress = min(
            float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
            1.0  # Hard clamp to prevent overshooting
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine decay
        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay  # No division by initial_lr
        return decayed_lr / initial_lr  # Normalize to [0, 1] for LambdaLR
    return LambdaLR(optimizer, lr_lambda)

def main():
    # --- MODIFIED: Add argument for resuming ---
    # parser = argparse.ArgumentParser(description="VP-VAE Training with Accelerate")
    # parser.add_argument("--resume_from_checkpoint", type=str, default=None,
    #                     help="Path to a checkpoint directory to resume training from.")
    # args = parser.parse_args()
    


    # # Initialize training state variables
    # # These will be overwritten if resuming from checkpoint
    # start_epoch = 0
    # completed_steps = 0 # Renamed from global_step to avoid conflict with loop var
    # best_eval_loss = float('inf')
    # # Initialize history lists
    # logged_steps_hist = []; step_lr_hist = []
    # step_loss_total_hist = []; step_loss_ce_elem_hist = []; step_loss_ce_cmd_hist = []
    # step_loss_mse_cont_hist = []; step_loss_kl_hist = []
    # step_eval_loss_hist = []

    accelerator = Accelerator(log_with="wandb")

    # IMPORTANT: DATASET_FILE must contain data in the new hybrid format
    # Col 0: Element ID (int), Col 1: Command ID (int), Cols 2-13: Continuous normalized params
    DATASET_FILE = os.path.join('/home/svgfusion_v2/', 'dataset/', 'optimized_progressive_dataset_precomputed_v8.pt') # EXAMPLE
    DATASET_FILE = os.path.abspath(DATASET_FILE)
    
    if accelerator.is_main_process: print(f"Loading HYBRID data from '{DATASET_FILE}'...")
    # ... (Rest of data loading logic as before) ...
    try:
        precomputed_data_list = torch.load(DATASET_FILE)
        if not isinstance(precomputed_data_list, list): raise TypeError("Loaded data not a list.")
        if accelerator.is_main_process and not precomputed_data_list: print("Warning: Loaded precomputed_data_list is empty!")
        if accelerator.is_main_process: print(f"Loaded {len(precomputed_data_list)} preprocessed SVG structures.")
    except Exception as e:
        if accelerator.is_main_process: print(f"Error loading data: {e}"); traceback.print_exc(); exit()


    # --- Determine dimensions (num_other_continuous_features) ---
    dino_embed_dim_from_data = 0
    num_other_continuous_features = 0 # This is N_geo + N_style
    if precomputed_data_list:
        first_item = precomputed_data_list[0]
        # For hybrid, first 2 cols are IDs, rest are continuous
        num_total_cols = first_item['full_svg_matrix_content'].shape[1]
        num_other_continuous_features = num_total_cols - 2 
        dino_embed_dim_from_data = first_item['cumulative_pixel_CLS_tokens_aligned'].shape[-1]
    else: # Fallbacks
        num_other_continuous_features = 12 # e.g., 8 geo + 4 style
        dino_embed_dim_from_data = 384
    
    # Get vocab sizes for categorical features from SVGToTensor_Normalized
    temp_converter = SVGToTensor_Normalized() # Assumes this is properly configured
    num_element_types = len(temp_converter.ELEMENT_TYPES)
    num_command_types = len(temp_converter.PATH_COMMAND_TYPES)
    # Padding IDs for embeddings/loss
    element_pad_idx = temp_converter.ELEMENT_TYPES.get('<PAD>', 0) # Default to 0 if <PAD> not explicitly in ELEMENT_TYPES
    command_pad_idx = temp_converter.PATH_COMMAND_TYPES.get('NO_CMD', 0) # Default to 0 for NO_CMD


    if accelerator.is_main_process: 
        print(f"Data: Num Element Types: {num_element_types}, Num Command Types: {num_command_types}, Num Other Cont. Feats: {num_other_continuous_features}, DINO Dim: {dino_embed_dim_from_data}")
        print(f"Accelerate: Number of processes: {accelerator.num_processes}")

    config_dict = {
        "learning_rate": 3e-4, "total_steps": 5000, "batch_size_per_device": 32,
        "warmup_steps": 100, "lr_decay_min": 1.5e-5, "weight_decay": 0.1,
        "log_interval": 10, "eval_interval": 100,
        "latent_dim": 128, "encoder_layers": 4, "decoder_layers": 4,
        "encoder_d_model": 256, "decoder_d_model": 256,
        "element_embed_dim": 64, "command_embed_dim": 64,
        "num_other_continuous_svg_features": num_other_continuous_features, # For encoder and decoder
        "pixel_feature_dim": dino_embed_dim_from_data, "num_heads": 8, 
        "kl_weight_max": 1e-6, "kl_anneal_portion": 0.8,
        "ce_elem_weight": 1.0, "ce_cmd_weight": 1.5, "mse_cont_weight": 10.0, # Adjusted weights
        "dataset_source": DATASET_FILE, "architecture": "VPVAE_Accel_HybridLoss_NoTau",
        "max_seq_len_train": 1024, 
        "num_element_types": num_element_types, "num_command_types": num_command_types,
        "element_padding_idx": element_pad_idx, "command_padding_idx": command_pad_idx,
        "seed": 42
    }
    
    # ... (WandB init, set_seed, code logging as before) ...
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="vp-vae-ce-loss", config=config_dict) # New project name
        wandb_run_name = "default_run"
        if hasattr(accelerator.get_tracker("wandb"), "run") and accelerator.get_tracker("wandb").run is not None:
            wandb_run_name = accelerator.get_tracker("wandb").run.name
    set_seed(config_dict["seed"], accelerator=accelerator)
    max_seq_len_for_model_and_dataset = config_dict["max_seq_len_train"]

    # SOS/EOS/PAD Token Rows for HYBRID data (elem_id, cmd_id, then continuous normalized)
    # IDs should be floats for tensor cat, but will be cast to long before embedding lookup
    # Continuous parts should be normalized defaults.
    defult_geo_values = torch.full((num_other_continuous_features-4,), temp_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)
    default_style_values = torch.full((4,), temp_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)
    #default_cont_param_values = torch.full((num_other_continuous_features,), temp_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)
    default_cont_param_values = torch.cat([defult_geo_values, default_style_values], dim=0)

    sos_elem_id_val = float(temp_converter.ELEMENT_TYPES['<BOS>'])
    eos_elem_id_val = float(temp_converter.ELEMENT_TYPES['<EOS>'])
    pad_elem_id_val = float(element_pad_idx) # Use determined pad_idx

    no_cmd_id_val = float(command_pad_idx) # Use NO_CMD as padding for command type

    sos_token_row = torch.cat([torch.tensor([sos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values])
    eos_token_row = torch.cat([torch.tensor([eos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values])
    padding_row_template = torch.cat([torch.tensor([pad_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values])
    
    sos_pixel_embed=torch.zeros((1,config_dict["pixel_feature_dim"]),dtype=torch.float32)
    pixel_padding_template=torch.zeros((1,config_dict["pixel_feature_dim"]),dtype=torch.float32)

    full_dataset = DynamicProgressiveSVGDataset(
        precomputed_data_list=precomputed_data_list, max_seq_length=max_seq_len_for_model_and_dataset,
        sos_token_row=sos_token_row, eos_token_row=eos_token_row, padding_row_template=padding_row_template,
        sos_pixel_embed=sos_pixel_embed, pixel_padding_template=pixel_padding_template)
    if len(full_dataset) == 0: print("ERROR: Dynamic dataset empty!"); exit()

    # ... (Train/Eval split, DataLoader creation as before) ...
    eval_share = 0.01; num_eval_samples = max(config_dict["batch_size_per_device"] * accelerator.num_processes * 2, int(len(full_dataset) * eval_share)) 
    num_eval_samples = min(num_eval_samples, len(full_dataset) - (len(full_dataset) % accelerator.num_processes if len(full_dataset) >= accelerator.num_processes else 0) )
    if num_eval_samples < accelerator.num_processes * config_dict["batch_size_per_device"] and len(full_dataset) >= accelerator.num_processes * config_dict["batch_size_per_device"] : 
        num_eval_samples = accelerator.num_processes * config_dict["batch_size_per_device"]
    num_eval_samples = min(num_eval_samples, len(full_dataset))
    indices = list(range(len(full_dataset))); random.shuffle(indices) 
    if num_eval_samples > 0 and len(full_dataset) > num_eval_samples :
        train_indices = indices[:-num_eval_samples]; eval_indices = indices[-num_eval_samples:]
        train_dataset = Subset(full_dataset, train_indices); eval_dataset_subset = Subset(full_dataset, eval_indices)
        if accelerator.is_main_process: print(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset_subset)} eval.")
    else: 
        train_dataset = full_dataset; eval_dataset_subset = full_dataset
        if accelerator.is_main_process: print(f"Using full dataset for train/eval ({len(full_dataset)} samples).")
    def collate_fn(batch):
        svg_matrices = torch.stack([item[0] for item in batch]); pixel_embeds = torch.stack([item[1] for item in batch]); attention_masks = torch.stack([item[2] for item in batch])
        return svg_matrices, pixel_embeds, attention_masks
    train_dataloader = DataLoader(train_dataset, batch_size=config_dict["batch_size_per_device"], shuffle=True, num_workers=4, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset_subset, batch_size=config_dict["batch_size_per_device"], shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = VPVAE(
        num_element_types=config_dict["num_element_types"],
        num_command_types=config_dict["num_command_types"],
        element_embed_dim=config_dict["element_embed_dim"],
        command_embed_dim=config_dict["command_embed_dim"],
        num_other_continuous_svg_features=config_dict["num_other_continuous_svg_features"],
        num_other_continuous_params_to_reconstruct=config_dict["num_other_continuous_svg_features"], # Assuming same number for reconstruction
        pixel_feature_dim=config_dict["pixel_feature_dim"],
        encoder_d_model=config_dict["encoder_d_model"], 
        decoder_d_model=config_dict["decoder_d_model"],
        encoder_layers=config_dict["encoder_layers"], 
        decoder_layers=config_dict["decoder_layers"],
        num_heads=config_dict["num_heads"], 
        latent_dim=config_dict["latent_dim"], 
        max_seq_len=max_seq_len_for_model_and_dataset,
        element_padding_idx=config_dict["element_padding_idx"],
        command_padding_idx=config_dict["command_padding_idx"]
    )

    optimizer = optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=config_dict["weight_decay"], betas=(0.9, 0.999))
    steps_after_warmup = config_dict["total_steps"] - config_dict["warmup_steps"]
    #scheduler = CosineAnnealingLR(optimizer, T_max=max(1, steps_after_warmup), eta_min=config_dict["lr_decay_min"])
    scheduler = cosine_warmup_scheduler(
                    optimizer,
                    warmup_steps=config_dict["warmup_steps"] * 2,
                    total_steps=config_dict["total_steps"] * 2,
                    initial_lr=config_dict["learning_rate"],
                    min_lr=config_dict["lr_decay_min"]
                )
    
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler)
    
    if accelerator.is_main_process: print("\nStarting Training (Hybrid Loss, No Tau Regression)...") # Updated print

    # --- Training Loop (needs to unpack hybrid outputs and use hybrid loss) ---
    logged_steps = []; step_loss_history = {'total': [], 'ce_elem': [], 'ce_cmd': [], 'mse_cont': [], 'kl': []} # New loss keys
    # ... (rest of history init)
    step_lr_history = []; step_eval_loss_history = []
    running_losses = {'total': 0.0, 'ce_elem':0.0, 'ce_cmd':0.0, 'mse_cont': 0.0, 'kl': 0.0} # New loss keys
    global_step = 0; best_eval_loss = float('inf')

    num_epochs = math.ceil(config_dict["total_steps"] / len(train_dataloader))
    if accelerator.is_main_process: print(f"Training for ~{num_epochs} epochs to reach ~{config_dict['total_steps']} steps.")

    for epoch in range(num_epochs):
        model.train()
        epoch_pbar = tqdm(train_dataloader, desc=f"E{epoch+1}", disable=not accelerator.is_main_process, position=0, leave=True)
        for batch_data in epoch_pbar:
            if global_step >= config_dict["total_steps"]: break
            svg_matrices_hybrid_batch, img_embeds_batch, svg_mask_batch = batch_data
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # Model now returns hybrid outputs
                elem_logits, cmd_logits, cont_params_pred, mu, log_var = model(
                    svg_matrices_hybrid_batch, img_embeds_batch, svg_mask_batch, svg_mask_batch
                )
                current_kl_weight = get_kl_weight(global_step, config_dict["total_steps"], config_dict["kl_weight_max"], config_dict["kl_anneal_portion"])
                
                loss, ce_e, ce_c, mse_p, kl = vp_vae_hybrid_loss(
                    elem_logits, cmd_logits, cont_params_pred,
                    svg_matrices_hybrid_batch, mu, log_var,
                    element_pad_idx=config_dict["element_padding_idx"], 
                    command_pad_idx=config_dict["command_padding_idx"],
                    svg_padding_mask=svg_mask_batch, 
                    kl_weight=current_kl_weight,
                    ce_elem_loss_weight=config_dict["ce_elem_weight"],
                    ce_cmd_loss_weight=config_dict["ce_cmd_weight"],
                    mse_cont_loss_weight=config_dict["mse_cont_weight"]
                )
                if torch.isnan(loss) or torch.isinf(loss): # ... (NaN check)
                    if accelerator.is_main_process: print(f"NaN/Inf (GS{global_step}). Skip.");
                    optimizer.zero_grad(); continue
                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            
            scheduler.step() # Update scheduler after optimizer step
            current_lr = scheduler.get_last_lr()[0] # Get current learning rate
            # current_lr = optimizer.param_groups[0]['lr'] 
            # # ... (LR schedule update logic as before) ...
            # # if global_step < config_dict["warmup_steps"]:
            # #     lr_scale = float(global_step + 1) / float(config_dict["warmup_steps"])
            # #     scaled_lr = config_dict["learning_rate"] * lr_scale
            # #     for param_group in optimizer.param_groups: param_group['lr'] = scaled_lr
            # #     current_lr = scaled_lr
            # # elif global_step == config_dict["warmup_steps"] and accelerator.is_main_process: 
            # #     print(f"Warmup done. LR: {optimizer.param_groups[0]['lr']:.2e}")
            # # elif global_step > config_dict["warmup_steps"]: 
            # #      scheduler.step(); current_lr = scheduler.get_last_lr()[0]
            # if global_step < config_dict["warmup_steps"]:
            #     # Manual linear warmup
            #     lr_scale = float(global_step + 1) / float(max(1, config_dict["warmup_steps"]))
            #     scaled_lr = config_dict["learning_rate"] * lr_scale 
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = scaled_lr
            #     current_lr = scaled_lr # LR used for this step
            
            # else: # global_step >= warmup_steps (Scheduler takes over)
            #     # This 'else' block will be entered when global_step == warmup_steps for the first time
            #     if global_step == config_dict["warmup_steps"] and accelerator.is_main_process:
            #         # Optimizer LR is at its peak from the last warmup step (global_step = warmup_steps - 1).
            #         # Now, scheduler.step() will be called for global_step = warmup_steps.
            #         print(f"Warmup phase complete at END of GS={global_step-1}. Optimizer LR is {optimizer.param_groups[0]['lr']:.2e}. Scheduler now active for GS={global_step}.")
                
            #     # Important: Ensure optimizer has the correct starting LR for the scheduler 
            #     # *before* the first scheduler.step() if it wasn't set by the last warmup iteration.
            #     # In our case, at global_step == warmup_steps, the LR in optimizer *should* be config_dict["learning_rate"]
            #     # because the last warmup scaling happened at global_step = warmup_steps - 1.
                
            #     if accelerator.is_main_process:
            #         scheduler.step()
            #     accelerator.wait_for_everyone() 
            #     current_lr = optimizer.param_groups[0]['lr']
            
            # Update running losses with new components
            running_losses['total']+=loss.item(); running_losses['ce_elem']+=ce_e.item(); 
            running_losses['ce_cmd']+=ce_c.item(); running_losses['mse_cont']+=mse_p.item();
            running_losses['kl']+=kl.item()
            global_step += 1
            
            if accelerator.is_main_process:
                desc = f"E{epoch+1} GS{global_step} L:{loss.item():.3f}(E:{ce_e.item():.3f},C:{ce_c.item():.3f},P:{mse_p.item():.3f},KL:{kl.item():.3f}) LR:{current_lr:.5f}"
                epoch_pbar.set_description(desc)
                if global_step % config_dict["log_interval"] == 0:
                    avg_losses = {k: v / config_dict["log_interval"] for k, v in running_losses.items()}
                    log_data = {"step": global_step, "epoch": epoch + 1, 
                                "avg_loss_total": avg_losses['total'], # Renamed for clarity
                                "avg_loss_ce_elem": avg_losses['ce_elem'],
                                "avg_loss_ce_cmd": avg_losses['ce_cmd'],
                                "avg_loss_mse_cont": avg_losses['mse_cont'],
                                "avg_loss_kl_div": avg_losses['kl'], # Renamed for clarity
                                "current_kl_weight": current_kl_weight, "learning_rate": current_lr,
                                "mu_mean": mu.mean().item(), "mu_std": mu.std().item()}
                    accelerator.log(log_data, step=global_step)
                    logged_steps.append(global_step); step_lr_history.append(current_lr)
                    for key_hist in step_loss_history: step_loss_history[key_hist].append(avg_losses[key_hist.replace("mse_recon","mse_cont")]) # Adjust key
                    running_losses = {k: 0.0 for k in running_losses}
                
            # --- Evaluation Block (Modified for Hybrid Loss) ---
            if global_step % config_dict["eval_interval"] == 0 or global_step == config_dict["total_steps"]:
                model.eval()
                eval_local_scalars = {'total': [], 'ce_elem': [], 'ce_cmd': [], 'mse_cont': [], 'kl': []}
                
                eval_pbar = tqdm(eval_dataloader, desc=f"R{accelerator.process_index} Eval GS{global_step}", 
                                 disable=False, position=accelerator.process_index, leave=True)
                for eval_batch_data in eval_pbar:
                    try:
                        eval_svg, eval_pix, eval_mask = eval_batch_data
                        with torch.no_grad():
                            el_lg, cmd_lg, cont_pred, mu_eval, lv_eval = model(eval_svg, eval_pix, eval_mask, eval_mask)
                            ev_loss, ev_ce_e, ev_ce_c, ev_mse_p, ev_kl = vp_vae_hybrid_loss(
                                el_lg, cmd_lg, cont_pred, eval_svg, mu_eval, lv_eval,
                                config_dict["element_padding_idx"], config_dict["command_padding_idx"],
                                eval_mask, current_kl_weight,
                                config_dict["ce_elem_weight"], config_dict["ce_cmd_weight"], config_dict["mse_cont_weight"]
                            )
                        eval_local_scalars['total'].append(ev_loss.item())
                        eval_local_scalars['ce_elem'].append(ev_ce_e.item())
                        eval_local_scalars['ce_cmd'].append(ev_ce_c.item())
                        eval_local_scalars['mse_cont'].append(ev_mse_p.item())
                        eval_local_scalars['kl'].append(ev_kl.item())
                    except Exception as e_eval_batch: # ... (error handling as before)
                        accelerator.print(f"!!! Rank {accelerator.process_index} Eval ERROR: {e_eval_batch} !!!")
                        traceback.print_exc(file=sys.stderr); sys.stderr.flush()
                        for k_ls in eval_local_scalars: eval_local_scalars[k_ls].append(float('nan'))

                # Stack collected scalars into a single tensor per process
                # [num_local_eval_batches, num_loss_components (5)]
                if eval_local_scalars['total']:
                    local_stacked_losses = torch.tensor(
                        [eval_local_scalars['total'], eval_local_scalars['ce_elem'], eval_local_scalars['ce_cmd'], 
                         eval_local_scalars['mse_cont'], eval_local_scalars['kl']],
                        device=accelerator.device, dtype=torch.float32
                    ).transpose(0,1) # Transpose to get [N_batches, 5]
                else:
                    local_stacked_losses = torch.empty(0, 5, device=accelerator.device, dtype=torch.float32)
                
                accelerator.wait_for_everyone()
                gathered_losses_tensor = accelerator.gather_for_metrics(local_stacked_losses)
                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    avg_eval_metrics = {}
                    loss_keys_ordered = ['total', 'ce_elem', 'ce_cmd', 'mse_cont', 'kl']
                    if gathered_losses_tensor.numel() > 0 and gathered_losses_tensor.shape[0] > 0:
                        valid_gathered = gathered_losses_tensor[~torch.isnan(gathered_losses_tensor).any(dim=1)]
                        if valid_gathered.numel() > 0 and valid_gathered.shape[0] > 0:
                            means = valid_gathered.mean(dim=0)
                            for idx, key_suffix in enumerate(loss_keys_ordered):
                                avg_eval_metrics[f'avg_eval_{key_suffix}'] = means[idx].item()
                        else: # All NaN or empty after filter
                            for key_suffix in loss_keys_ordered: avg_eval_metrics[f'avg_eval_{key_suffix}'] = float('nan')
                    else: # Gathered tensor empty
                        for key_suffix in loss_keys_ordered: avg_eval_metrics[f'avg_eval_{key_suffix}'] = float('nan')
                    
                    accelerator.print(f"Main proc GS {global_step}: Eval Metrics: {avg_eval_metrics}")
                    accelerator.log(avg_eval_metrics, step=global_step)
                    # ... (rest of saving best model logic, using avg_eval_metrics['avg_eval_total']) ...
                    current_eval_total_loss = avg_eval_metrics.get('avg_eval_total', float('inf'))
                    step_eval_loss_history.append(current_eval_total_loss)
                    if not np.isnan(current_eval_total_loss) and current_eval_total_loss < best_eval_loss:
                        best_eval_loss = current_eval_total_loss
                        unwrapped_model = accelerator.unwrap_model(model)
                        current_run_name_for_save = wandb_run_name if 'wandb_run_name' in locals() and wandb_run_name else "unknown_run"
                        save_path_model_state = f"vp_vae_accel_hybrid_{current_run_name_for_save}_s{global_step}_best.pt"
                        accelerator.save(unwrapped_model.state_dict(), save_path_model_state)
                        accelerator.print(f"Best eval: {best_eval_loss:.4f}. Saved: {save_path_model_state}")


                accelerator.wait_for_everyone()
                model.train()
        # ... (end of training batch loop and epoch loop) ...
    # ... (final plotting logic, update keys for step_loss_history) ...
    if accelerator.is_main_process:
        print("\nTraining Finished."); accelerator.end_training()
        if logged_steps: 
            print("\nGenerating training plots...")
            plt.figure(figsize=(24, 10)) # Made figure larger
            plt.subplot(2,3,1); plt.plot(logged_steps, step_loss_history['total'], label='Total'); plt.title('Avg Total Loss'); plt.legend(); plt.grid(True);plt.ylim(bottom=0)
            plt.subplot(2,3,2); plt.plot(logged_steps, step_loss_history['ce_elem'], label='CE Elem'); plt.title('Avg CE Elem Loss'); plt.legend(); plt.grid(True);plt.ylim(bottom=0)
            plt.subplot(2,3,3); plt.plot(logged_steps, step_loss_history['ce_cmd'], label='CE Cmd'); plt.title('Avg CE Cmd Loss'); plt.legend(); plt.grid(True);plt.ylim(bottom=0)
            plt.subplot(2,3,4); plt.plot(logged_steps, step_loss_history['mse_cont'], label='MSE Cont'); plt.title('Avg MSE Cont Loss'); plt.legend(); plt.grid(True);plt.ylim(bottom=0)
            plt.subplot(2,3,5); plt.plot(logged_steps, step_loss_history['kl'], label='KL'); plt.title('Avg KL (Unweighted)'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)
            
            eval_plot_steps = [s for s in range(0, config_dict["total_steps"] + 1, config_dict["eval_interval"])][1:] 
            valid_eval_history = [l for l in step_eval_loss_history if not np.isnan(l)] 
            if len(eval_plot_steps) > len(valid_eval_history): eval_plot_steps = eval_plot_steps[:len(valid_eval_history)]
            elif len(eval_plot_steps) < len(valid_eval_history) and valid_eval_history: valid_eval_history = valid_eval_history[:len(eval_plot_steps)]
            if eval_plot_steps and valid_eval_history : 
                plt.subplot(2,3,6); plt.plot(eval_plot_steps, valid_eval_history, label='Avg Eval Loss',marker='o');plt.title('Avg Eval Loss');plt.legend();plt.grid(True);plt.ylim(bottom=0)
            else: print("Not enough valid data for eval plot.")
            plt.tight_layout(); plt.savefig("vp_vae_accel_hybrid_training_curves.png"); print("Saved training curves.")


if __name__ == "__main__":
    main()
# --- END OF FILE vpvae_accelerate_ce.py ---
