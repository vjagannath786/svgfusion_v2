# --- START OF FILE vpvae_accelerate_eval_dl.py ---
import sys
sys.path.append('/kaggle/working')


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset # Added Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb
import math
import random
import os
import traceback

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed as accelerate_set_seed

from svgutils import DynamicProgressiveSVGDataset
from svgutils import SVGToTensor_Normalized

# --- Set Seed Utility ---
def set_seed(seed, accelerator=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if accelerator is not None: accelerate_set_seed(seed)
    else:
        if torch.backends.mps.is_available():
            try: torch.mps.manual_seed(seed)
            except AttributeError: pass

# --- RoPE Utility, Model Definitions (MultiHeadAttention, etc.) ---
# --- Assume these are THE SAME as your previous version ---
# --- (For brevity, I'm omitting them here, but they should be included) ---
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
    def __init__(self, num_total_svg_features, pixel_feature_dim, d_model, num_layers, num_heads, latent_dim, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len; self.d_model = d_model; self.svg_input_feature_dim = num_total_svg_features
        self.svg_projection = nn.Linear(self.svg_input_feature_dim, d_model); self.pixel_projection = nn.Linear(pixel_feature_dim, d_model)
        self.svg_norm_pre_proj = nn.LayerNorm(self.svg_input_feature_dim); self.pixel_norm_pre_proj = nn.LayerNorm(pixel_feature_dim)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, kdim=d_model, vdim=d_model, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.self_attention_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff=d_model*4, batch_first=True) for _ in range(num_layers)])
        self.fc_mu = nn.Linear(d_model, latent_dim); self.fc_var = nn.Linear(d_model, latent_dim)
    def forward(self, svg_matrix_continuous, pixel_embedding, svg_padding_mask=None, pixel_padding_mask=None):
        svg_features_rope = apply_rope(svg_matrix_continuous); svg_features_normed = self.svg_norm_pre_proj(svg_features_rope)
        svg_projected = self.svg_projection(svg_features_normed)
        pixel_features_normed = self.pixel_norm_pre_proj(pixel_embedding) 
        pixel_projected = self.pixel_projection(pixel_features_normed)
        cross_attn_output = self.cross_attention(query=svg_projected, key=pixel_projected, value=pixel_projected, key_padding_mask=pixel_padding_mask)
        x = self.cross_attn_norm(svg_projected + cross_attn_output)
        if svg_padding_mask is not None: x = x * (~svg_padding_mask).unsqueeze(-1).float()
        for layer in self.self_attention_layers: x = layer(x, padding_mask=svg_padding_mask)
        if svg_padding_mask is not None:
            valid_mask = (~svg_padding_mask).unsqueeze(-1).float()
            pooled_x = (x * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-9)
        else: pooled_x = torch.mean(x, dim=1)
        mu = self.fc_mu(pooled_x); log_var = self.fc_var(pooled_x)
        return mu, log_var

class VPVAEDecoder(nn.Module): 
    def __init__(self, num_total_svg_features_to_reconstruct, latent_dim, d_model, num_layers, num_heads, max_seq_len):
        super().__init__(); self.max_seq_len = max_seq_len; self.d_model = d_model
        self.num_total_svg_features_to_reconstruct = num_total_svg_features_to_reconstruct
        self.fc_latent = nn.Linear(latent_dim, d_model)
        self.decoder_layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff=d_model*4, batch_first=True) for _ in range(num_layers)])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_head_continuous = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_total_svg_features_to_reconstruct), nn.Tanh())
    def forward(self, z, target_len):
        batch_size=z.size(0); x=self.fc_latent(z).unsqueeze(1); x=x.repeat(1, target_len, 1)
        effective_target_len = min(target_len, self.max_seq_len); x_rope = apply_rope(x[:,:effective_target_len,:])
        causal_mask=torch.triu(torch.ones(effective_target_len, effective_target_len, device=z.device), diagonal=1).bool()
        for layer in self.decoder_layers: x_rope=layer(x_rope, padding_mask=None, attn_mask=causal_mask)
        x_normed=self.decoder_norm(x_rope); predicted_continuous_svg_features = self.output_head_continuous(x_normed)
        return predicted_continuous_svg_features

class VPVAE(nn.Module): 
    def __init__(self, num_total_svg_features, pixel_feature_dim, encoder_d_model, decoder_d_model, encoder_layers, decoder_layers, num_heads, latent_dim, max_seq_len):
        super().__init__()
        self.encoder=VPVAEEncoder(num_total_svg_features, pixel_feature_dim, encoder_d_model, encoder_layers, num_heads, latent_dim, max_seq_len)
        self.decoder=VPVAEDecoder(num_total_svg_features, latent_dim, decoder_d_model, decoder_layers, num_heads, max_seq_len)
        self.max_seq_len=max_seq_len
    def reparameterize(self, mu, logvar): std=torch.exp(0.5*torch.clamp(logvar,min=-10,max=10)); eps=torch.randn_like(std); return mu+eps*std
    def forward(self, svg_matrix_continuous, pixel_embedding, svg_padding_mask=None, pixel_padding_mask=None):
        mu,logvar=self.encoder(svg_matrix_continuous, pixel_embedding, svg_padding_mask, pixel_padding_mask)
        z=self.reparameterize(mu, logvar); predicted_continuous_svg_features = self.decoder(z, target_len=self.max_seq_len)
        return predicted_continuous_svg_features, mu, logvar

def vp_vae_loss_full_mse(predicted_continuous_svg_features, target_svg_matrix_continuous, mu, logvar,
                         svg_padding_mask=None, kl_weight=0.1, recon_mse_loss_weight=1.0):
    batch_size, target_len_out, num_features_out = predicted_continuous_svg_features.shape
    _, target_len_svg, num_features_svg = target_svg_matrix_continuous.shape
    assert num_features_out == num_features_svg, "Feature dimension mismatch"
    effective_len = min(target_len_out, target_len_svg)
    preds_eff = predicted_continuous_svg_features[:, :effective_len, :]
    targets_eff = target_svg_matrix_continuous[:, :effective_len, :]
    if svg_padding_mask is not None:
        svg_padding_mask_eff = svg_padding_mask[:, :effective_len]
        feature_valid_mask = (~svg_padding_mask_eff).unsqueeze(-1).expand_as(targets_eff).float()
        num_valid_elements = feature_valid_mask.sum() + 1e-9
    else:
        feature_valid_mask = torch.ones_like(targets_eff)
        num_valid_elements = targets_eff.numel()
    mse_loss_recon_unreduced = F.mse_loss(preds_eff * feature_valid_mask, targets_eff * feature_valid_mask, reduction='none')
    mse_loss_recon = mse_loss_recon_unreduced.sum() / num_valid_elements
    logvar_exp_clamped = torch.clamp(logvar, max=80.0) 
    kl_div=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar_exp_clamped.exp(), dim=1)
    kl_loss=torch.mean(kl_div)
    total_loss = recon_mse_loss_weight * mse_loss_recon + kl_weight * kl_loss
    return total_loss, mse_loss_recon, kl_loss

def get_kl_weight(step, total_steps, max_kl_weight=0.1, anneal_portion=0.8, schedule='linear'):
    anneal_steps = int(total_steps * anneal_portion)
    if step < anneal_steps: return max_kl_weight * (step / max(1, anneal_steps))
    else: return max_kl_weight
# =============================================================================

def main():
    accelerator = Accelerator(log_with="wandb")

    DATASET_FILE = os.path.join('/kaggle/input/', 'progressive/', 'optimized_progressive_dataset_precomputed.pt')
    DATASET_FILE = os.path.abspath(DATASET_FILE)
    
    if accelerator.is_main_process: print(f"Loading data from '{DATASET_FILE}'...")
    if not os.path.exists(DATASET_FILE):
        if accelerator.is_main_process: print(f"ERROR: Dataset file '{DATASET_FILE}' not found."); exit()
    try:
        precomputed_data_list = torch.load(DATASET_FILE)
        if not isinstance(precomputed_data_list, list): raise TypeError("Loaded data not a list.")
        if accelerator.is_main_process and not precomputed_data_list: print("Warning: Loaded precomputed_data_list is empty!")
        if accelerator.is_main_process: print(f"Loaded {len(precomputed_data_list)} preprocessed SVG structures.")
    except Exception as e:
        if accelerator.is_main_process: print(f"Error loading data: {e}"); traceback.print_exc(); exit()

    num_total_svg_features = 0; dino_embed_dim_from_data = 0
    if precomputed_data_list:
        first_item = precomputed_data_list[0]
        num_total_svg_features = first_item['full_svg_matrix_content'].shape[1] if 'full_svg_matrix_content' in first_item else 15
        dino_embed_dim_from_data = first_item['cumulative_pixel_CLS_tokens_aligned'].shape[-1] if 'cumulative_pixel_CLS_tokens_aligned' in first_item else 384
    else: num_total_svg_features = 15; dino_embed_dim_from_data = 384
    if accelerator.is_main_process: print(f"Data: SVG Feats: {num_total_svg_features}, DINO Dim: {dino_embed_dim_from_data}")

    config_dict = {
        "learning_rate": 3e-4, "total_steps": 5000, "batch_size_per_device": 32,
        "warmup_steps": 100, "lr_decay_min": 1.5e-6, "weight_decay": 0.1,
        "log_interval": 10, "eval_interval": 100, # Increased eval interval
        "latent_dim": 256, "encoder_layers": 4, "decoder_layers": 4,
        "encoder_d_model": 512, "decoder_d_model": 512,
        "pixel_feature_dim": dino_embed_dim_from_data, "num_heads": 8, 
        "kl_weight_max": 0.005, "kl_anneal_portion": 0.6, # Start KL very low
        "recon_mse_loss_weight": 1.0, "dataset_source": DATASET_FILE, 
        "architecture": "VPVAE_Accel_DynProg_EvalDL_FullMSE",
        "max_seq_len_train": 1024, "num_total_svg_features": num_total_svg_features, "seed": 42
    }
    
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="vp-vae-dynamic-progressive-accelerate", config=config_dict)
        wandb_run_name = "default_run"
        if hasattr(accelerator.get_tracker("wandb"), "run") and accelerator.get_tracker("wandb").run is not None:
            wandb_run_name = accelerator.get_tracker("wandb").run.name
            
    set_seed(config_dict["seed"], accelerator=accelerator)
    max_seq_len_for_model_and_dataset = config_dict["max_seq_len_train"]

    # Code logging (main process only)
    if accelerator.is_main_process:
        # ... (code logging logic as before, ensure wandb_run_name is available) ...
        try:
            current_run_name = wandb_run_name if 'wandb_run_name' in locals() and wandb_run_name else "unknown_run"
            code_artifact = wandb.Artifact(name=f'source-code-{current_run_name}',type='code', description='VP-VAE Accel DynProg EvalDL Full MSE',metadata=config_dict)
            code_artifact.add_file(__file__)
            # Add other files robustly
            base_dir = os.path.dirname(__file__)
            for f_name in ['dataset_preparation_dynamic.py', 'svgtensor.py']: # Assuming these are in svgutils
                path_to_check = os.path.abspath(os.path.join(base_dir, '..', 'svgutils', f_name))
                if os.path.exists(path_to_check): code_artifact.add_file(path_to_check, name=f'svgutils/{f_name}')
                else: print(f"Could not find {path_to_check} for artifact logging.")
            wandb.log_artifact(code_artifact); print("Logged source code via WandB artifact (main process).")
        except Exception as e: print(f"Warning: Failed to log code artifact (main process): {e}")


    temp_svg_tensor_converter = SVGToTensor_Normalized()
    # ... (SOS/EOS/PAD token row generation as before, ensure it's correct) ...
    element_min_val=min(temp_svg_tensor_converter.ELEMENT_TYPES.values());element_max_val=max(temp_svg_tensor_converter.ELEMENT_TYPES.values())
    cmd_type_min_val=min(temp_svg_tensor_converter.PATH_COMMAND_TYPES.values());cmd_type_max_val=max(temp_svg_tensor_converter.PATH_COMMAND_TYPES.values())
    cmd_seq_idx_min_val=getattr(temp_svg_tensor_converter,'cmd_seq_idx_min',0.0);cmd_seq_idx_max_val=getattr(temp_svg_tensor_converter,'cmd_seq_idx_max',100.0)
    norm_sos_elem=temp_svg_tensor_converter._normalize(temp_svg_tensor_converter.ELEMENT_TYPES['<BOS>'],element_min_val,element_max_val)
    norm_eos_elem=temp_svg_tensor_converter._normalize(temp_svg_tensor_converter.ELEMENT_TYPES['<EOS>'],element_min_val,element_max_val)
    norm_pad_elem=temp_svg_tensor_converter._normalize(temp_svg_tensor_converter.ELEMENT_TYPES['<PAD>'],element_min_val,element_max_val)
    norm_zero_cmd_seq=temp_svg_tensor_converter._normalize(0.0,cmd_seq_idx_min_val,cmd_seq_idx_max_val)
    norm_no_cmd_type=temp_svg_tensor_converter._normalize(temp_svg_tensor_converter.PATH_COMMAND_TYPES['NO_CMD'],cmd_type_min_val,cmd_type_max_val)
    default_other_params=torch.full((temp_svg_tensor_converter.num_geom_params+temp_svg_tensor_converter.num_fill_style_params,),temp_svg_tensor_converter.DEFAULT_PARAM_VAL,dtype=torch.float32)
    sos_token_row=torch.cat([torch.tensor([norm_sos_elem,norm_zero_cmd_seq,norm_no_cmd_type],dtype=torch.float32),default_other_params])
    eos_token_row=torch.cat([torch.tensor([norm_eos_elem,norm_zero_cmd_seq,norm_no_cmd_type],dtype=torch.float32),default_other_params])
    padding_row_template=torch.cat([torch.tensor([norm_pad_elem,norm_zero_cmd_seq,norm_no_cmd_type],dtype=torch.float32),default_other_params])
    sos_pixel_embed=torch.zeros((1,config_dict["pixel_feature_dim"]),dtype=torch.float32)
    pixel_padding_template=torch.zeros((1,config_dict["pixel_feature_dim"]),dtype=torch.float32)


    full_dataset = DynamicProgressiveSVGDataset(
        precomputed_data_list=precomputed_data_list,max_seq_length=max_seq_len_for_model_and_dataset,
        sos_token_row=sos_token_row,eos_token_row=eos_token_row,padding_row_template=padding_row_template,
        sos_pixel_embed=sos_pixel_embed,pixel_padding_template=pixel_padding_template)
    if len(full_dataset) == 0: print("ERROR: Dynamic dataset empty!"); exit()

    # --- MODIFIED: Create a dedicated evaluation subset and DataLoader ---
    eval_share = 0.01 # Use 1% for validation, or a fixed number
    num_eval_samples = max(config_dict["batch_size_per_device"] * accelerator.num_processes * 2, int(len(full_dataset) * eval_share)) # At least 2 batches
    
    # Ensure we don't try to take more samples than available, or too few for all processes
    num_eval_samples = min(num_eval_samples, len(full_dataset) - (len(full_dataset) % accelerator.num_processes if len(full_dataset) >= accelerator.num_processes else 0) )
    if num_eval_samples < accelerator.num_processes and len(full_dataset) >= accelerator.num_processes : # need at least one sample per process for eval
        num_eval_samples = accelerator.num_processes * config_dict["batch_size_per_device"]
    num_eval_samples = min(num_eval_samples, len(full_dataset))


    # Create indices for train and eval splits
    # Shuffling indices before splitting is good practice if data isn't already random
    indices = list(range(len(full_dataset)))
    random.shuffle(indices) # Shuffle once for consistent splits across runs with same seed

    if num_eval_samples > 0 and len(full_dataset) > num_eval_samples :
        train_indices = indices[:-num_eval_samples]
        eval_indices = indices[-num_eval_samples:]
        train_dataset = Subset(full_dataset, train_indices)
        eval_dataset_subset = Subset(full_dataset, eval_indices)
        if accelerator.is_main_process:
            print(f"Split dataset: {len(train_dataset)} train samples, {len(eval_dataset_subset)} eval samples.")
    else: # Not enough data for a split, use all for training and evaluation (less ideal)
        train_dataset = full_dataset
        eval_dataset_subset = full_dataset # Or a small fixed subset for quicker eval checks
        if accelerator.is_main_process:
             print(f"Using full dataset for both training and evaluation due to small size ({len(full_dataset)} samples).")


    def collate_fn(batch):
        svg_matrices = torch.stack([item[0] for item in batch])
        pixel_embeds = torch.stack([item[1] for item in batch])
        attention_masks = torch.stack([item[2] for item in batch])
        return svg_matrices, pixel_embeds, attention_masks

    train_dataloader = DataLoader(train_dataset, batch_size=config_dict["batch_size_per_device"], shuffle=True, num_workers=0, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset_subset, batch_size=config_dict["batch_size_per_device"], shuffle=False, num_workers=0, collate_fn=collate_fn)


    model = VPVAE(
        num_total_svg_features=config_dict["num_total_svg_features"], pixel_feature_dim=config_dict["pixel_feature_dim"],
        encoder_d_model=config_dict["encoder_d_model"], decoder_d_model=config_dict["decoder_d_model"],
        encoder_layers=config_dict["encoder_layers"], decoder_layers=config_dict["decoder_layers"],
        num_heads=config_dict["num_heads"], latent_dim=config_dict["latent_dim"], max_seq_len=max_seq_len_for_model_and_dataset)

    optimizer = optim.AdamW(model.parameters(), lr=config_dict["learning_rate"], weight_decay=config_dict["weight_decay"], betas=(0.9, 0.95))
    steps_after_warmup = config_dict["total_steps"] - config_dict["warmup_steps"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, steps_after_warmup), eta_min=config_dict["lr_decay_min"]) # T_max should be > 0
    
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    ) # Prepare eval_dataloader too
    
    if accelerator.is_main_process:
        print("\nStarting VP-VAE Training (Accelerate, DynProg, EvalDL, FullMSE)...")
        print(f"Device: {accelerator.device}, Num Procs: {accelerator.num_processes}, Dist Type: {accelerator.distributed_type}")
        print(f"Effective batch size: {config_dict['batch_size_per_device'] * accelerator.num_processes}")

    logged_steps = []; step_loss_history = {'total': [], 'mse_recon': [], 'kl': []}
    step_lr_history = []; step_eval_loss_history = []
    running_losses = {'total': 0.0, 'mse_recon': 0.0, 'kl': 0.0}
    global_step = 0; best_eval_loss = float('inf')
    
    num_epochs = math.ceil(config_dict["total_steps"] / len(train_dataloader))
    if accelerator.is_main_process: print(f"Training for ~{num_epochs} epochs to reach ~{config_dict['total_steps']} steps.")

    for epoch in range(num_epochs):
        model.train()
        epoch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process, position=0, leave=True)
        for batch_data in epoch_pbar:
            if global_step >= config_dict["total_steps"]: break
            svg_matrices_batch_cont, img_embeds_batch, svg_mask_batch = batch_data
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predicted_svg_features, mu, log_var = model(svg_matrices_batch_cont, img_embeds_batch, svg_mask_batch, svg_mask_batch)
                current_kl_weight = get_kl_weight(global_step, config_dict["total_steps"], config_dict["kl_weight_max"], config_dict["kl_anneal_portion"])
                loss, mse_recon_loss, kl_loss = vp_vae_loss_full_mse(predicted_svg_features, svg_matrices_batch_cont, mu, log_var, svg_mask_batch, current_kl_weight, config_dict["recon_mse_loss_weight"])
                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process: print(f"NaN/Inf (GS{global_step}). Skip.");
                    optimizer.zero_grad(); continue
                accelerator.backward(loss)
                if accelerator.sync_gradients: accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            current_lr = optimizer.param_groups[0]['lr'] 
            if global_step < config_dict["warmup_steps"]:
                lr_scale = float(global_step + 1) / float(config_dict["warmup_steps"])
                scaled_lr = config_dict["learning_rate"] * lr_scale
                for param_group in optimizer.param_groups: param_group['lr'] = scaled_lr
                current_lr = scaled_lr
            elif global_step == config_dict["warmup_steps"] and accelerator.is_main_process: 
                print(f"Warmup done. LR: {optimizer.param_groups[0]['lr']:.2e}")
            if global_step >= config_dict["warmup_steps"]: # Scheduler step after optimizer step
                 scheduler.step(); current_lr = scheduler.get_last_lr()[0]
            
            running_losses['total']+=loss.item(); running_losses['mse_recon']+=mse_recon_loss.item(); running_losses['kl']+=kl_loss.item()
            global_step += 1
            
            if accelerator.is_main_process:
                epoch_pbar.set_description(f"E{epoch+1} GS{global_step} L:{loss.item():.3f}(MSE:{mse_recon_loss.item():.3f},KL:{kl_loss.item():.3f}) LR:{current_lr:.1e}")
                if global_step % config_dict["log_interval"] == 0:
                    avg_losses = {k: v / config_dict["log_interval"] for k, v in running_losses.items()}
                    log_data = {"step": global_step, "epoch": epoch + 1, "avg_loss_interval": avg_losses['total'],
                                "avg_mse_recon_loss_interval": avg_losses['mse_recon'], "avg_kl_div_interval": avg_losses['kl'],
                                "current_kl_weight": current_kl_weight, "learning_rate": current_lr,
                                "mu_mean": mu.mean().item(), "mu_std": mu.std().item()}
                    accelerator.log(log_data, step=global_step)
                    logged_steps.append(global_step); step_lr_history.append(current_lr)
                    for key_hist in step_loss_history: step_loss_history[key_hist].append(avg_losses[key_hist])
                    running_losses = {k: 0.0 for k in running_losses}

                
            if global_step % config_dict["eval_interval"] == 0 or global_step == config_dict["total_steps"]:
                    model.eval()
                    # These will store lists of Python floats (scalars) from this process
                    local_total_scalars = []
                    local_mse_scalars = []
                    local_kl_scalars = []

                    try:
                        current_eval_dl_len = len(eval_dataloader)
                        accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: My eval_dataloader (prepared) length: {current_eval_dl_len}")
                    except Exception as e_len_dl:
                        accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: ERROR getting len(eval_dataloader): {e_len_dl}")
                    


                    
                    
                    eval_pbar_desc = f"Rank {accelerator.process_index} Evaluating GS{global_step}"
                    eval_pbar = tqdm(
                        eval_dataloader, 
                        desc=eval_pbar_desc,
                        disable=False,
                        position=accelerator.process_index, # accelerator.process_index if you want to see all
                        leave=True
                    )

                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: Starting iteration over my eval_dataloader shard.")
                    for eval_batch_idx, eval_batch_data in enumerate(eval_pbar): # eval_pbar is per-process now
                        accelerator.print(f"------------ Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: Top of loop. ------------")
                        try:
                            eval_svg, eval_pix, eval_mask = eval_batch_data
                            
                            accelerator.print(f"Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: Data loaded, SVG shape {eval_svg.shape}")
                            with torch.no_grad():
                                accelerator.print(f"Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: Before model forward.")
                                pred_svg, mu_eval, lv_eval = model(
                                    eval_svg, eval_pix, eval_mask, eval_mask
                                )
                                accelerator.print(f"Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: After model forward. Pred_svg shape {pred_svg.shape}")
                                
                                ev_loss, ev_mse_r, ev_kl = vp_vae_loss_full_mse(
                                    pred_svg, eval_svg, mu_eval, lv_eval,
                                    eval_mask, current_kl_weight,
                                    config_dict["recon_mse_loss_weight"]
                                )
                                accelerator.print(f"Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: After loss. Loss val: {ev_loss.item():.4f}")
                            
                            local_total_scalars.append(ev_loss.detach().clone())
                            local_mse_scalars.append(ev_mse_r.detach().clone())
                            local_kl_scalars.append(ev_kl.detach().clone())
                            accelerator.print(f"Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: Appended losses.")

                        except Exception as e_batch_eval:
                            accelerator.print(f"!!!!!!!! Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: ERROR during eval batch: {e_batch_eval} !!!!!!!!")
                            traceback.print_exc(file=sys.stderr) # This will print the traceback for the failing rank
                            print(f"--- TRACEBACK Rank {accelerator.process_index} ---", file=sys.stderr)
                            sys.stderr.flush()
                            accelerator.print(f"Rank {accelerator.process_index} is ABORTING due to error in eval.")
                            
                            # Crucially, ensure this error causes the process to exit or signal others.
                            # For now, let's make it explicitly exit to make the failure clear.
                            # Or, if we want to debug gather, we need all processes to reach gather.
                            # So, append NaNs and continue for now, but be aware an error occurred.
                            placeholder_loss = torch.tensor(float('nan'), device=accelerator.device)
                            local_total_scalars.append(placeholder_loss)
                            local_mse_scalars.append(placeholder_loss)
                            local_kl_scalars.append(placeholder_loss)
                            accelerator.print(f"Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: Appended NaNs due to error.")
                            #raise e_batch_eval
                            # To make it stop immediately on error on any rank:
                            # accelerator.set_trigger() # Fictional function, but idea is to signal other procs
                            # raise e_batch_eval # This might not be caught well by accelerate's top level sometimes
                        accelerator.print(f"------------ Rank {accelerator.process_index} GS {global_step} EvalBatch {eval_batch_idx}: Bottom of loop. ------------")
                    
                    
                    # This is where Rank 0 gets to, but Rank 1 might not if it's stuck or errored above.
                    #accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: Finished iterating MY eval_dataloader. Num local batches processed: {eval_batch_idx + 1 if 'eval_batch_idx' in locals() and eval_batch_idx >=0 else 0}. Local losses collected (counts): { {k: len(v) for k, v in eval_local_losses_collected.items()} } items.")

                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: Finished MY eval_dataloader. Batches processed: {len(local_total_scalars)}.")
                    
                    # Convert collected scalar lists to a single tensor for this process
                    # Each tensor will be [num_local_eval_batches_on_this_proc, 1]
                    # Then stack to [num_local_eval_batches_on_this_proc, 3]
                    # if local_total_scalars: # If any batches were processed
                    #     losses_to_gather_total = torch.tensor(local_total_scalars, device=accelerator.device).unsqueeze(1)
                    #     losses_to_gather_mse = torch.tensor(local_mse_scalars, device=accelerator.device).unsqueeze(1)
                    #     losses_to_gather_kl = torch.tensor(local_kl_scalars, device=accelerator.device).unsqueeze(1)
                    #     # Stack them: each process now has a tensor of shape [N_local_batches, 3]
                    #     local_stacked_eval_losses = torch.cat([losses_to_gather_total, losses_to_gather_mse, losses_to_gather_kl], dim=1)
                    # else: # This process had no eval batches (should be rare if eval_dataset is not tiny)
                    #     local_stacked_eval_losses = torch.empty(0, 3, device=accelerator.device, dtype=torch.float32)

                    if local_total_scalars: # If any batches were processed
                        losses_to_gather_total = torch.stack(local_total_scalars).unsqueeze(1)
                        losses_to_gather_mse = torch.stack(local_mse_scalars).unsqueeze(1)
                        losses_to_gather_kl = torch.stack(local_kl_scalars).unsqueeze(1)
                        # Stack them: each process now has a tensor of shape [N_local_batches, 3]
                        local_stacked_eval_losses = torch.cat([losses_to_gather_total, losses_to_gather_mse, losses_to_gather_kl], dim=1)
                    else:
                        local_stacked_eval_losses = torch.empty(0, 3, device=accelerator.device, dtype=torch.float32)

                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: Local stacked losses shape: {local_stacked_eval_losses.shape}. Waiting for sync before gather.")
                    accelerator.wait_for_everyone() # Sync before gather
                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: SYNCED. Calling gather_for_metrics.")
                    
                    gathered_losses_tensor = accelerator.gather_for_metrics(local_stacked_eval_losses) # Should be [TotalEvalBatchesAcrossAllProcs, 3]
                    
                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: Gathered losses. Shape: {gathered_losses_tensor.shape}. Waiting for sync after gather.")
                    accelerator.wait_for_everyone() # Sync after gather
                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: SYNCED after gather.")
                    
                    # Calculate and log metrics (only on main process)
                    if accelerator.is_main_process:
                        avg_eval_metrics = {}
                        if gathered_losses_tensor.numel() > 0 and gathered_losses_tensor.shape[0] > 0:
                            # Filter NaNs that might have resulted from errors on other processes
                            valid_gathered_losses = gathered_losses_tensor[~torch.isnan(gathered_losses_tensor).any(dim=1)]
                            if valid_gathered_losses.numel() > 0 and valid_gathered_losses.shape[0] > 0:
                                avg_eval_metrics['avg_eval_total'] = valid_gathered_losses[:, 0].mean().item()
                                avg_eval_metrics['avg_eval_mse_recon'] = valid_gathered_losses[:, 1].mean().item()
                                avg_eval_metrics['avg_eval_kl'] = valid_gathered_losses[:, 2].mean().item()
                            else:
                                accelerator.print("WARN: All gathered eval losses were NaN or list was empty after filtering.")
                                for k_metric in ['total', 'mse_recon', 'kl']: avg_eval_metrics[f'avg_eval_{k_metric}'] = float('nan')
                        else:
                            accelerator.print("WARN: Gathered eval losses tensor is empty.")
                            for k_metric in ['total', 'mse_recon', 'kl']: avg_eval_metrics[f'avg_eval_{k_metric}'] = float('nan')
                        
                        accelerator.print(f"Main proc GS {global_step}: Calculated final metrics: {avg_eval_metrics}")
                        accelerator.print(f"\n--- GS{global_step} Eval Summary (Main Process)|L:{avg_eval_metrics.get('avg_eval_total',0):.3f}(MSE:{avg_eval_metrics.get('avg_eval_mse_recon',0):.3f},KL:{avg_eval_metrics.get('avg_eval_kl',0):.3f}) ---")
                        accelerator.log(avg_eval_metrics, step=global_step)
                        current_eval_total_loss = avg_eval_metrics.get('avg_eval_total', float('inf'))
                        step_eval_loss_history.append(current_eval_total_loss)

                        if not np.isnan(current_eval_total_loss) and current_eval_total_loss < best_eval_loss:
                            best_eval_loss = current_eval_total_loss
                            
                            unwrapped_model = accelerator.unwrap_model(model)
                            current_run_name_for_save = wandb_run_name if 'wandb_run_name' in locals() and wandb_run_name else "unknown_run"
                            save_path_model_state = f"vp_vae_accel_dynprog_{current_run_name_for_save}_s{global_step}_best.pt"
                            accelerator.save(unwrapped_model.state_dict(), save_path_model_state)
                            accelerator.print(f"Best eval: {best_eval_loss:.4f}. Saved: {save_path_model_state}")
                    
                    accelerator.wait_for_everyone() # Sync before main process saves model
                    model.train() # Set back to train mode
                    accelerator.print(f"Rank {accelerator.process_index} GS {global_step}: Set model to train. End of eval block.")
        if global_step >= config_dict["total_steps"]: break
        if accelerator.is_main_process: epoch_pbar.close()

    if accelerator.is_main_process:
        print("\nTraining Finished."); accelerator.end_training()
        if logged_steps: # Plotting ... (as before)
            print("\nGenerating training plots...")
            plt.figure(figsize=(20, 5))
            plt.subplot(1,3,1); plt.plot(logged_steps, step_loss_history['total'], label='Total'); plt.plot(logged_steps, step_loss_history['mse_recon'], label='MSE Recon'); plt.title('Avg Losses'); plt.legend(); plt.grid(True);plt.ylim(bottom=0)
            plt.subplot(1,3,2); plt.plot(logged_steps, step_loss_history['kl'], label='KL'); plt.title('Avg KL (Unweighted)'); plt.legend(); plt.grid(True); plt.ylim(bottom=0)
            eval_plot_steps = [s for s in range(0, config_dict["total_steps"] + 1, config_dict["eval_interval"])][1:] 
            valid_eval_history = [l for l in step_eval_loss_history if not np.isnan(l)] # Filter NaNs for plotting
            if len(eval_plot_steps) > len(valid_eval_history): eval_plot_steps = eval_plot_steps[:len(valid_eval_history)]
            elif len(eval_plot_steps) < len(valid_eval_history) and valid_eval_history: valid_eval_history = valid_eval_history[:len(eval_plot_steps)]
            if eval_plot_steps and valid_eval_history : 
                plt.subplot(1,3,3); plt.plot(eval_plot_steps, valid_eval_history, label='Avg Eval Loss',marker='o');plt.title('Avg Eval Loss');plt.legend();plt.grid(True);plt.ylim(bottom=0)
            else: print("Not enough valid data for eval plot.")
            plt.tight_layout(); plt.savefig("vp_vae_accel_dynprog_training_curves.png"); print("Saved training curves.")


if __name__ == "__main__":
    main()
# --- END OF FILE vpvae_accelerate_eval_dl.py ---