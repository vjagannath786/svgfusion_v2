# --- START OF FILE generate_samples.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import math
import random
import os
import gc
import sys
import traceback
from pathlib import Path
import matplotlib.pyplot as plt

# Adjust sys.path for your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust this if your models/svgutils are elsewhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'svgutils')))


# --- Import VAE Components ---
try:
    from models import VPVAE # Assuming VPVAE definition is in vpvae_accelerate_ce.py
    print("Successfully imported VAE components.")
except ImportError as e:
    print(f"Error importing VPVAE from models.vpvae_accelerate_ce: {e}. Ensure the file exists and VPVAE class is defined.")
    traceback.print_exc()
    sys.exit(1)

# --- Import VS-DiT Model and Diffusion Utils (Sequence Conditioning Version) ---
try:
    from models import VS_DiT, get_linear_noise_schedule, precompute_diffusion_parameters, ddim_sample
    print("Successfully imported VS-DiT (SeqCond) model and diffusion utilities.")
except ImportError as e:
    print(f"Error importing VS_DiT model/utils from models.vp_dit: {e}. Ensure the file exists and classes are defined.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during import of DiT model/utils: {e}"); traceback.print_exc(); sys.exit(1)


# --- Import SVG Reconstruction ---
try:
    from svgutils import tensor_to_svg_file_hybrid_wrapper, SVGToTensor_Normalized
    print("Successfully imported SVG reconstruction utility.")
except ImportError as e:
    print(f"Error importing SVG reconstruction utility from svgutils.py: {e}. Ensure the file exists and classes are defined.")
    traceback.print_exc()
    sys.exit(1)

# --- Set Seed Utility ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try: torch.mps.manual_seed(seed)
        except AttributeError: pass





# -----------------------------------------------------------------------------
# Configuration and Setup
# -----------------------------------------------------------------------------

# --- Paths ---
# !!! ADJUST THESE PATHS TO YOUR ACTUAL CHECKPOINTS AND CLIP MODEL !!!
PATH_VAE_CHECKPOINT = '/workspace/best-model/vp_vae_accel_hybrid_fallen-jazz-36_s13500_best.pt' # VAE trained with sequential latents
PATH_VSDIT_CHECKPOINT = "/workspace/saved_models_vsdit_square_overfit/vsdit_clip_seqcond_classic-armadillo-96_best.pth" # DiT trained with sequential latents
PATH_CLIP_MODEL = "/workspace/clip-model-large-vit/clip-vit-large-patch14" # HuggingFace model name or local path
OUTPUT_DIR = "./generated_svgs_vsdit_seqcond_sequential_latent" # New output dir

# --- Model & Sampling Parameters ---
# These will be largely INFERRED from the loaded checkpoints for consistency.
# Define sensible defaults here that will be overridden/verified.
DEFAULT_LATENT_FEATURE_DIM = 32 # Based on your previous discussion
DEFAULT_VAE_MAX_SEQ_LEN_SVG_CMDS = 1024 # Standard N
DEFAULT_VSDIT_HIDDEN_DIM = 384 # Default d_model
DEFAULT_VSDIT_CONTEXT_DIM = 768 # CLIP output dim
DEFAULT_VSDIT_NUM_BLOCKS = 12
DEFAULT_VSDIT_NUM_HEADS = 6
DEFAULT_VSDIT_MLP_RATIO = 4.0
DEFAULT_VSDIT_DROPOUT = 0.0

DIFFUSION_NUM_TIMESTEPS = 1000
SAMPLING_NUM_STEPS_DDIM = 100 # Fewer steps for faster generation
SAMPLING_CFG_SCALE = 1.0 # CRITICAL FIX: Use 1.0 (Pure Conditional) because we trained with dropout=0.0
SAMPLING_ETA = 0.0 # 0.0 for deterministic DDIM

# --- Setup Device ---
if torch.cuda.is_available(): device = torch.device("cuda") # Use cuda if available, otherwise cpu
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(42) # Set seed for reproducibility


# In generate_samples_v1.py, near the top configuration section
PATH_ZDATASET = "/workspace/z-list-pt/z_latents_file_list.pt" # Path to the dataset object

# Load the dataset to get normalization stats
print(f"Loading zDataset from {PATH_ZDATASET} to get normalization stats...")
try:
    # Load zDataset class definition first
    from models import zDataset
    #full_zdataset = torch.load(PATH_ZDATASET)
    full_zdataset = zDataset(z_file_list_path=PATH_ZDATASET)
    z_mean = full_zdataset.z_mean.to(device)
    z_std = full_zdataset.z_std.to(device)
    print("Successfully loaded z_mean and z_std.")
except Exception as e:
    print(f"FATAL: Could not load zDataset for un-normalization stats: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Load VAE (Ensure it's the version outputting sequential latents)
# -----------------------------------------------------------------------------
print(f"Loading VAE checkpoint from '{PATH_VAE_CHECKPOINT}'...")
try:
    checkpoint_vae = torch.load(PATH_VAE_CHECKPOINT, map_location='cpu')
    
    # Infer VAE config from checkpoint if possible, or use defaults
    # A more robust solution would be to save the config alongside the model.
    # For now, we manually construct a config that *must match training*.
    # Use SVGToTensor_Normalized to get vocab sizes consistently.
    temp_svg_tensor_converter_for_vae_config = SVGToTensor_Normalized() 
    _num_element_types = len(temp_svg_tensor_converter_for_vae_config.ELEMENT_TYPES)
    _num_command_types = len(temp_svg_tensor_converter_for_vae_config.PATH_COMMAND_TYPES)
    #_num_other_continuous_svg_features = temp_svg_tensor_converter_for_vae_config.num_geom_params + temp_svg_tensor_converter_for_vae_config.num_fill_style_params
    _num_other_continuous_svg_features = 12
    _element_padding_idx = temp_svg_tensor_converter_for_vae_config.ELEMENT_TYPES.get('<PAD>', 0) 
    _command_padding_idx = temp_svg_tensor_converter_for_vae_config.PATH_COMMAND_TYPES.get('NO_CMD', 0)

    # VAE config - these values MUST match how your VAE was TRAINED
    # They are hardcoded here but ideally loaded from a saved config.
    config = {
        "max_seq_len_train": DEFAULT_VAE_MAX_SEQ_LEN_SVG_CMDS,
        "num_element_types": _num_element_types,
        "num_command_types": _num_command_types,
        "element_embed_dim": 64,  # e.g., 64
        "command_embed_dim": 64,  # e.g., 64
        "num_other_continuous_svg_features": _num_other_continuous_svg_features, # e.g., 12
        "pixel_feature_dim": 384, # e.g., DINOv2-small output
        "encoder_d_model": 512, 
        "decoder_d_model": 512, 
        "encoder_layers": 4, 
        "decoder_layers": 4, 
        "num_heads": 8, 
        "latent_dim": 32, # THIS IS CRUCIAL: Must match VAE's latent_dim
        "element_padding_idx": _element_padding_idx,
        "command_padding_idx": _command_padding_idx,
        "num_bins": 256 # From training config
    }
    
    print(config)

    full_vae_model = VPVAE( num_element_types=config["num_element_types"],
        num_command_types=config["num_command_types"],
        element_embed_dim=config["element_embed_dim"],
        command_embed_dim=config["command_embed_dim"],
        num_other_continuous_svg_features=config["num_other_continuous_svg_features"],
        num_other_continuous_params_to_reconstruct=config["num_other_continuous_svg_features"],
        pixel_feature_dim=config["pixel_feature_dim"],
        encoder_d_model=config["encoder_d_model"], 
        decoder_d_model=config["decoder_d_model"],
        encoder_layers=config["encoder_layers"], 
        decoder_layers=config["decoder_layers"],
        num_heads=config["num_heads"], 
        latent_dim=config["latent_dim"], 
        max_seq_len=config["max_seq_len_train"],
        element_padding_idx=config["element_padding_idx"],
        command_padding_idx=config["command_padding_idx"] )
    full_vae_model.load_state_dict(checkpoint_vae)
    print("Full VAE (sequential latent version) weights loaded.")
    
    vae_decoder = full_vae_model.decoder.to(device).eval()
    
    # Store relevant VAE config values for later use
    VAE_MAX_SEQ_LEN_SVG_CMDS = config["max_seq_len_train"]
    LATENT_FEATURE_DIM = config["latent_dim"] # Confirming latent_feature_dim from VAE
    VAE_NUM_OTHER_CONT_FEATURES = config["num_other_continuous_svg_features"]

    print("VAE Decoder (sequential latent version) extracted and set to eval mode.")
    del full_vae_model, checkpoint_vae; gc.collect()
except Exception as e: print(f"Error loading VAE checkpoint: {e}"); traceback.print_exc(); sys.exit(1)

# -----------------------------------------------------------------------------
# Load VS-DiT Model (Ensure it's version for sequential latents)
# -----------------------------------------------------------------------------
print("Instantiating VS-DiT model (for Sequential Latents)...")
try:
    vs_dit_state_dict = torch.load(PATH_VSDIT_CHECKPOINT, map_location='cpu') # Load to CPU first

    # VS-DiT config - these values MUST match how your VS-DiT was TRAINED
    # They are hardcoded here but ideally loaded from a saved config.
    vs_dit_config = {
        "latent_dim": LATENT_FEATURE_DIM, # Input feature dim per token, confirmed from VAE
        "hidden_dim": DEFAULT_VSDIT_HIDDEN_DIM, 
        "context_dim": DEFAULT_VSDIT_CONTEXT_DIM, # CLIP output dim
        "num_blocks": DEFAULT_VSDIT_NUM_BLOCKS, 
        "num_heads": DEFAULT_VSDIT_NUM_HEADS, 
        "mlp_ratio": DEFAULT_VSDIT_MLP_RATIO, 
        "dropout": DEFAULT_VSDIT_DROPOUT 
    }

    vs_dit_model = VS_DiT(**vs_dit_config)
    vs_dit_model.load_state_dict(vs_dit_state_dict)
    vs_dit_model.to(device).eval() # Move to device and set to eval mode
    print(f"VS-DiT (sequential latent version) weights loaded from '{PATH_VSDIT_CHECKPOINT}'.")
except Exception as e: print(f"Error loading VS-DiT state dict: {e}"); traceback.print_exc(); sys.exit(1)

# -----------------------------------------------------------------------------
# Load CLIP Model
# -----------------------------------------------------------------------------
print(f"Loading CLIP model from {PATH_CLIP_MODEL}...")
try:
    clip_model = CLIPTextModel.from_pretrained(PATH_CLIP_MODEL).to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained(PATH_CLIP_MODEL) # Use AutoTokenizer for flexibility
    clip_model.eval()
    clip_output_dim = clip_model.config.hidden_size
    if vs_dit_config["context_dim"] != clip_output_dim:
         print(f"FATAL ERROR: VS-DiT context dim ({vs_dit_config['context_dim']}) != CLIP last_hidden_state dim ({clip_output_dim}).")
         sys.exit(1)
    print("CLIP model loaded successfully.")
except Exception as e: print(f"Error loading CLIP: {e}"); traceback.print_exc(); sys.exit(1)

# -----------------------------------------------------------------------------
# Load Diffusion Parameters (relates to timesteps T)
# -----------------------------------------------------------------------------
print("Setting up diffusion parameters...")
betas = get_linear_noise_schedule(DIFFUSION_NUM_TIMESTEPS)
diff_params = precompute_diffusion_parameters(betas, device) # For DDIM
print(f"Diffusion parameters precomputed for {DIFFUSION_NUM_TIMESTEPS} steps.")


# -----------------------------------------------------------------------------
# Generation Function
# -----------------------------------------------------------------------------
def generate_svg_from_prompt(
    prompt: str,
    vs_dit_model: nn.Module,
    vae_decoder: nn.Module,
    clip_model: nn.Module,
    clip_tokenizer: object,
    device: torch.device,
    num_samples: int = 1,
    cfg_scale: float = 3.0,
    ddim_diff_params: dict = None,
    ddim_steps: int = 1000,
    ddim_eta: float = 0.0,
    # Inferred parameters
    latent_seq_len: int = VAE_MAX_SEQ_LEN_SVG_CMDS, # Now directly derived from VAE config
    latent_feature_dim: int = LATENT_FEATURE_DIM,   # Now directly derived from VAE config
    vae_target_svg_len: int = VAE_MAX_SEQ_LEN_SVG_CMDS,
    vae_num_params_svg: int = VAE_NUM_OTHER_CONT_FEATURES, # From VAE config
    latent_min_max: tuple = (-10, 10), # Default clamping
    latent_mask: torch.Tensor = None # Optional mask
):
    print(f"\n--- Generating {num_samples} sample(s) for prompt: '{prompt}' ---")

    # 2. Encode Text Prompts
    print(f"Encoding prompt: '{prompt}'")
    text_inputs = clip_tokenizer(
        [prompt] * num_samples,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        clip_outputs = clip_model(**text_inputs)
        context_seq = clip_outputs.last_hidden_state
        pooled_context = clip_outputs.pooler_output # [B, D_clip]
        context_padding_mask = ~(text_inputs.attention_mask.bool())

    # 3. Sample Latents (DDIM)
    print(f"Sampling latents with DDIM (Steps={ddim_steps}, CFG={cfg_scale})...")
    
    # Determine shape: [B, N, latent_dim]
    if hasattr(vs_dit_model, 'proj_in'):
         latent_dim = vs_dit_model.proj_in.in_features
    else:
         latent_dim = 32 # Default fallback
         
    num_svg_tokens = 1024 # Standard for this project
    
    shape = (num_samples, num_svg_tokens, latent_dim)
    
    # Run DDIM Sampling
    sampled_latents, _ = ddim_sample(
        model=vs_dit_model,
        shape=shape,
        context_seq=context_seq,
        pooled_context=pooled_context, # Pass pooled context
        context_padding_mask=context_padding_mask,
        diff_params=ddim_diff_params,
        num_timesteps=1000, # Force 1000 for now to match training
        target_device=device,
        cfg_scale=cfg_scale,
        eta=ddim_eta,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        return_visuals=False,
        latent_min_max=latent_min_max,
        ddim_steps=ddim_steps # Pass ddim_steps
    )
    # sampled_latents: [num_samples, LATENT_SEQ_LEN, LATENT_FEATURE_DIM]
    print(f"sampled latents shape: {sampled_latents.shape} | Mean: {sampled_latents.mean():.4f}, Std: {sampled_latents.std():.4f}")
    
    
    # 2. Un-normalize Latents
    # 2. Un-normalize Latents
    # Load stats from file
    try:
        z_mean = torch.load("/workspace/z_mean_banana_overfit.pt", map_location=device)
        z_std = torch.load("/workspace/z_std_banana_overfit.pt", map_location=device)
        print(f"Loaded Circle Stats for Sampling -> Mean Shape: {z_mean.shape}, Std Shape: {z_std.shape}")
        print(f"Global Mean: {z_mean.mean().item():.4f}, Global Std: {z_std.mean().item():.4f}")
    except Exception as e:
        print(f"ERROR: Could not load circle stats: {e}. Cannot proceed with accurate sampling.")
        sys.exit(1)

    # Ensure shapes for broadcasting [1, 1, 32] -> [B, N, D]
    # z_mean/std are [32]
    new_mean = z_mean.view(1, 1, -1)
    new_std = z_std.view(1, 1, -1)
    
    # No GT mask available for general sampling
    gt_latent_mask = None
    gt_latent_padding_mask = None

    # ------------------------------
    
    unnormalized_latents = sampled_latents * new_std + new_mean
    #unnormalized_latents = sampled_latents
    
    # --- CRITICAL FIX: Zero out padding tokens ---
    # The model preserves noise in the padding region (because loss is masked).
    # We must remove this noise before decoding, otherwise the VAE tries to decode it.
    if gt_latent_mask is not None:
        # gt_latent_mask is [1, 1024] (1=Valid, 0=Padding)
        # Expand to [B, 1024, 32]
        mask_expanded = gt_latent_mask.unsqueeze(-1).expand_as(unnormalized_latents)
        # Apply mask: Keep valid, zero out padding
        unnormalized_latents = unnormalized_latents * mask_expanded.float()
        print("Applied GT mask to zero out padding noise in latents.")
    # ---------------------------------------------
    
    # --- INSPECT VALID TOKENS ---
    if gt_latent_mask is not None:
        # gt_latent_mask is [1, 1024]
        valid_indices = gt_latent_mask.bool()
        
        # Extract valid tokens from generated latent
        # unnormalized_latents is [B, 1024, 32]
        # We look at the first sample for inspection
        gen_valid = unnormalized_latents[0][valid_indices[0]] # [Num_Valid, 32]
        print(f"Generated Valid Tokens Stats -> Mean: {gen_valid.mean():.4f}, Std: {gen_valid.std():.4f}, Max: {gen_valid.max():.4f}")
        
        # Extract valid tokens from GT (loaded later, but let's load it here for comparison)
        try:
            z_gt_check = torch.load("z_latents_data/banana.pt_step_0.pt", map_location=device)
            gt_valid = z_gt_check[valid_indices[0]]
            print(f"GT Valid Tokens Stats        -> Mean: {gt_valid.mean():.4f}, Std: {gt_valid.std():.4f}, Max: {gt_valid.max():.4f}")
            
            # --- AMPLITUDE CORRECTION (VALID TOKENS ONLY) ---
            current_std = gen_valid.std()
            target_std = gt_valid.std()
            
            if current_std > 0:
                correction_factor = target_std / current_std
                print(f"Valid Token Correction Factor: {correction_factor:.4f}")
                
                if abs(correction_factor - 1.0) > 0.05:
                    # Apply scaling to valid tokens in the full tensor
                    # We need to do this for each sample in the batch
                    for b in range(unnormalized_latents.shape[0]):
                        # Get valid mask for this sample (assuming same mask for all if overfit)
                        mask_b = valid_indices[0] # [1024]
                        
                        # Get valid tokens
                        valid_tokens = unnormalized_latents[b][mask_b]
                        
                        # Scale
                        # Centering might be safer: (x - mean) * scale + mean
                        # But mean is close to 0. Let's just scale.
                        valid_tokens_scaled = valid_tokens * correction_factor
                        
                        # Put back
                        unnormalized_latents[b][mask_b] = valid_tokens_scaled
                    
                    print("Applied amplitude correction to valid tokens.")
            # ------------------------------------------------
            
        except Exception as e:
            print(f"Could not load GT for correction: {e}")
            pass

    print(f"Un-normalized latents shape: {unnormalized_latents.shape} | Mean: {unnormalized_latents.mean():.4f}, Std: {unnormalized_latents.std():.4f}")

    # --- COMPARISON WITH GROUND TRUTH ---
    # 5. Compare with Ground Truth (if available)
    # Load the specific square file we overfitted on
    try:
        gt_path = "z_latents_data/banana.pt_step_0.pt" # Adjust path if needed
        if os.path.exists(gt_path):
            z_gt = torch.load(gt_path, map_location=device) # [1024, 32]
            
            # Normalize GT for comparison
            z_gt_norm = (z_gt - z_mean) / z_std
            
            # Compare with first sample
            gen_z = unnormalized_latents[0] # [1024, 32]
            
            mse = F.mse_loss(gen_z, z_gt_norm).item()
            cosine = F.cosine_similarity(gen_z.flatten(), z_gt_norm.flatten(), dim=0).item()
            
            print(f"\n>>> MSE vs Ground Truth (Total): {mse:.6f} <<<")
            print(f">>> Cosine Similarity: {cosine:.6f} <<<")
            
            # NEW: Analyze First 14 Tokens (The Square) vs The Rest
            valid_len = 14
            mse_valid = F.mse_loss(gen_z[:valid_len], z_gt_norm[:valid_len]).item()
            mse_rest = F.mse_loss(gen_z[valid_len:], z_gt_norm[valid_len:]).item()
            
            print(f">>> MSE Valid ({valid_len} tokens): {mse_valid:.6f} <<<")
            print(f">>> MSE Padding (Rest): {mse_rest:.6f} <<<")
            
            print("\nFirst 5 GT Tokens (Dim 0):", z_gt_norm[:5, 0].tolist())
            print("First 5 Gen Tokens (Dim 0):", gen_z[:5, 0].tolist())
            
        else:
            print(f"Ground truth file not found at {gt_path}")
    except Exception as e:
        print(f"Could not compare with ground truth: {e}")
    # ------------------------------------
    
    #vae_latent_scale_factor = 0.4 
    #scaled_latents = unnormalized_latents * vae_latent_scale_factor
    
    #print(f"Final scaled latents shape: {scaled_latents.shape} | Mean: {scaled_latents.mean():.4f}, Std: {scaled_latents.std():.4f}")

    # 3. Decode Latents using VAE Decoder
    print("Decoding latents...")
    # 3. Decode with VAE
    # CRITICAL FIX: Latent Norm Truncation
    # The DiT generates zeros for padding, but the VAE decodes zeros to garbage.
    # We detect "zeros" by checking the norm and manually forcing the output to be PAD/EOS.
    
    # Calculate norms for each token in the batch
    latent_norms = unnormalized_latents.norm(dim=-1) # [B, 1024]
    is_padding_latent = latent_norms < 1.0 # Threshold: 1.0 (Valid norms are ~30)
    
    # --- SVGToTensor_Normalized (for getting PAD/EOS IDs consistently) ---
    # This needs to be defined before using pad_id
    svg_tensor_converter_decoder_ref = SVGToTensor_Normalized()
    _element_padding_idx = svg_tensor_converter_decoder_ref.ELEMENT_TYPES.get('<PAD>', 0) 
    _eos_id_int = svg_tensor_converter_decoder_ref.ELEMENT_TYPES.get('<EOS>', 0)
    pad_id = _element_padding_idx # Use the padding ID for overriding

    with torch.no_grad():
        vae_decoder.eval()
        # Use the VAE wrapper's decode method if available, or call decoder directly
        # Based on previous code, it seems we call vae.decoder directly or via wrapper
        # Let's assume vae_decoder is the VAE model or its decoder
        if hasattr(vae_decoder, 'decoder'):
             element_logits, command_logits, param_logits_list = vae_decoder.decoder(unnormalized_latents, target_len=VAE_MAX_SEQ_LEN_SVG_CMDS)
        else:
             element_logits, command_logits, param_logits_list = vae_decoder(unnormalized_latents, target_len=VAE_MAX_SEQ_LEN_SVG_CMDS)
    
    # 4. Convert logits to indices
    pred_elem_ids = element_logits.argmax(dim=-1) # [B, 1024]
    pred_cmds = command_logits.argmax(dim=-1)
    
    # OVERRIDE: If latent norm is low, force Element ID to PAD (or EOS)
    # We use the _element_padding_idx_for_final_save we got earlier
    pred_elem_ids[is_padding_latent] = pad_id
    
    pred_bin_indices_list = [
        torch.argmax(F.softmax(param_logits, dim=-1), dim=-1) # Use softmax for proper argmax
        for param_logits in param_logits_list
    ]
    pred_bin_indices = torch.stack(pred_bin_indices_list, dim=-1)
    
    combined_outputs = [] 
    batch_size_decode = unnormalized_latents.size(0) 
    for i in range(batch_size_decode):
        combined_output = torch.cat([
            pred_elem_ids[i].long().unsqueeze(-1), 
            pred_cmds[i].long().unsqueeze(-1), 
            pred_bin_indices[i].long()
        ], dim=-1) # combined_output: [VAE_MAX_SEQ_LEN_SVG_CMDS, num_svg_params_total]
        combined_outputs.append(combined_output)
    final_tensor_batch = torch.stack(combined_outputs, dim=0) # [B, VAE_MAX_SEQ_LEN_SVG_CMDS, num_svg_params_total]


    return final_tensor_batch

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    prompts_to_generate = [
        #"a white circle"
        # "a black circle"
        # "a blue diamond"
        #"a red square"
        #"a yellow square"
        #"a orange circle"
        # "a brown circle",
        #"a purple square"
        # "a green square"
        "A ripe yellow banana with the peel pulled back to reveal the fruit. The banana has a green stem and a small brown tip, designed in a simple flat style."
         
        # EXACT match from training data (including trailing space)
    ]
    n_samples_per_prompt = 2 # Generate 2 samples for each prompt

    svg_tensor_converter_instance = SVGToTensor_Normalized() # For final SVG reconstruction
    
    # --- Get PAD/EOS IDs from a consistent source (SVGToTensor_Normalized) ---
    # These are needed for actual_len detection in tensor_to_svg_file_hybrid_wrapper
    _element_padding_idx_for_final_save = svg_tensor_converter_instance.ELEMENT_TYPES.get('<PAD>', 0)
    _eos_id_int_for_final_save = svg_tensor_converter_instance.ELEMENT_TYPES.get('<EOS>', 0)

    # GT Mask loading removed for general sampling

    all_generated_tensors = []
    for prompt_text in prompts_to_generate:
        generated_tensors = generate_svg_from_prompt(
            prompt=prompt_text,
            vs_dit_model=vs_dit_model,
            vae_decoder=vae_decoder,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            device=device,
            num_samples=n_samples_per_prompt,
            cfg_scale=SAMPLING_CFG_SCALE,
            ddim_diff_params=diff_params,
            ddim_steps=SAMPLING_NUM_STEPS_DDIM,
            ddim_eta=SAMPLING_ETA,
            # Inferred parameters
            latent_seq_len=VAE_MAX_SEQ_LEN_SVG_CMDS,
            latent_feature_dim=LATENT_FEATURE_DIM,
            vae_target_svg_len=VAE_MAX_SEQ_LEN_SVG_CMDS,
            vae_num_params_svg=VAE_NUM_OTHER_CONT_FEATURES,
            latent_min_max=None, # Disable clamping to debug trajectory
            latent_mask=None # No mask for general sampling
        )
        all_generated_tensors.append(generated_tensors)

    if all_generated_tensors:
        final_tensor_batch = torch.cat(all_generated_tensors, dim=0)
        print(f"\nFinal combined tensor shape: {final_tensor_batch.shape}")
        print("\nSaving generated SVGs...")
        
        prompt_idx_save = 0
        sample_idx_save = 0
        for i in range(final_tensor_batch.size(0)):
            current_prompt = prompts_to_generate[prompt_idx_save]
            try:
                single_svg_tensor = final_tensor_batch[i] # [VAE_MAX_SEQ_LEN_SVG_CMDS, num_svg_params_total]
                safe_prompt_filename = current_prompt.replace(" ", "_").replace("/", "_").replace("'", "")[:100]
                output_filename = f'generated_{safe_prompt_filename}_s{sample_idx_save}.svg'
                output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                
                # Determine actual length (before padding or EOS) for reconstruction
                actual_recon_len = VAE_MAX_SEQ_LEN_SVG_CMDS # Start with max
                pred_elem_ids_list_cpu = single_svg_tensor[:,0].cpu().tolist() # Get element IDs for EOS/PAD check
                
                for i_len_rec in range(single_svg_tensor.shape[0]):
                    if pred_elem_ids_list_cpu[i_len_rec] == _eos_id_int_for_final_save or \
                       pred_elem_ids_list_cpu[i_len_rec] == _element_padding_idx_for_final_save:
                        actual_recon_len = i_len_rec 
                        break
                
                print(f"Reconstructing SVG with actual_len={actual_recon_len} for {output_filename}")
                
                print(single_svg_tensor[:10])
                tensor_to_svg_file_hybrid_wrapper(
                    single_svg_tensor,
                    output_filename=output_filepath,
                    svg_tensor_converter_instance=svg_tensor_converter_instance,
                    actual_len=actual_recon_len
                )
                print(f"Saved: {output_filepath}")
                
                sample_idx_save += 1
                if sample_idx_save >= n_samples_per_prompt:
                    sample_idx_save = 0
                    prompt_idx_save += 1
            except Exception as e:
                print(f"Error saving SVG for '{current_prompt}', sample {sample_idx_save}: {str(e)}")
                traceback.print_exc()
                sample_idx_save += 1 
                if sample_idx_save >= n_samples_per_prompt:
                    sample_idx_save = 0
                    prompt_idx_save += 1
    else:
        print("No tensors were generated.")

# --- END OF FILE generate_samples.py ---
