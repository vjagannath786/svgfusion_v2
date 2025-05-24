# --- START OF FILE generate_samples.py --- # Renamed

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import numpy as np
import math
import random
import os
import gc

# --- Import VAE Components ---
try:
    from models import (VPVAE, set_seed)
    print("Successfully imported VAE components.")
except ImportError as e:
    print(f"Error importing VAE components: {e}"); exit()

# --- Import VS-DiT Model and Diffusion Utils (Sequence Conditioning Version) ---
try:
    # !!! ENSURE THIS IS THE CORRECT FILENAME FOR THE SEQCOND MODEL !!!
    from models import VS_DiT # Use the correct class name
    from models import get_linear_noise_schedule, precompute_diffusion_parameters, ddim_sample
    print("Successfully imported VS-DiT (SeqCond) model and diffusion utilities.")
except ImportError as e:
    print(f"Error importing from test_vsdit_hidden_seqcond_CORRECTED.py: {e}"); exit()
except Exception as e:
     print(f"An unexpected error occurred during import: {e}"); exit()


# --- Import SVG Reconstruction ---
try:
    from svgutils import tensor_to_svg_file_hybrid_wrapper, SVGToTensor_Normalized
    print("Successfully imported SVG reconstruction utility.")
except ImportError as e:
    print(f"Error importing SVG reconstruction utility: {e}"); exit()




# -----------------------------------------------------------------------------
# Configuration and Setup
# -----------------------------------------------------------------------------

# --- Paths ---
# !!! ADJUST THESE PATHS !!!
PATH_VAE_CHECKPOINT = './best_models/vp_vae_accel_hybrid_good-eon-3_s19400_best.pt'
# --- MODIFIED: Path to the model trained with sequence conditioning ---
PATH_VSDIT_CHECKPOINT = "saved_models_vsdit_clip_seqcond/vsdit_clip_seqcond_revived-cosmos-6_best.pth" # Example name
PATH_CLIP_MODEL = "/Users/varun_jagannath/Documents/D/test python/clip-vit-large-patch14/" # Using CLIP-L
OUTPUT_DIR = "./generated_svgs_vsdit_seqcond" # New output dir

# --- Model & Sampling Parameters ---
# These MUST match the parameters used for training the loaded VS_DiT checkpoint
VSDIT_LATENT_DIM = 128
VSDIT_HIDDEN_DIM = 384 # Internal DiT dimension
# --- MODIFIED: Context dim must match LAST HIDDEN STATE ---
VSDIT_CONTEXT_DIM = 768    # CLIP ViT-L/14 last_hidden_state is 1024
VSDIT_NUM_BLOCKS = 12
VSDIT_NUM_HEADS = 6
VSDIT_MLP_RATIO = 8.0
VSDIT_DROPOUT = 0.0        # Set dropout to 0 for inference

DIFFUSION_NUM_TIMESTEPS = 1000 # Timesteps VS-DiT was trained with
SAMPLING_NUM_STEPS = 100    # Using full steps for sampling here
SAMPLING_CFG_SCALE = 3.0     # Classifier-Free Guidance scale
SAMPLING_ETA = 1.0           # Deterministic DDIM

# --- Dequantization Constants ---
ORIGINAL_PARAM_MIN_COORD = -127.0
STROKEW_MAX = 20.0
# VAE params loaded from checkpoint
PARAM_TYPES = { 0: 'coord', 1: 'coord', 2: 'coord', 3: 'coord', 4: 'rgb', 5: 'rgb', 6: 'rgb', 7: 'opacity', 8: 'rgb', 9: 'rgb'} # Adjust if needed

# --- Setup Device ---
# ... (device setup code - same as before) ...
if torch.cuda.is_available(): device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps") # Disabled MPS for stability if needed
else: device = torch.device("cpu")
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Load VAE Decoder (Unchanged from previous script)
# -----------------------------------------------------------------------------
print(f"Loading VAE checkpoint from '{PATH_VAE_CHECKPOINT}'...")
# ... (VAE loading code identical to previous script) ...
# (Ensures VAE_LATENT_DIM, NUM_BINS, NUM_PARAMS, MAX_SEQ_LEN are loaded from vae_config)

vp_vae_config = {
        "max_seq_len_train": 1024,
        "num_element_types": 7,
        "num_command_types": 14,
        "element_embed_dim": 64,  # Example, use actual from training
        "command_embed_dim": 64,  # Example, use actual from training
        "num_other_continuous_svg_features": 12,
        "pixel_feature_dim": 384, # Example, use actual (dino_embed_dim_from_data in train script)
        "encoder_d_model": 512,   # Example
        "decoder_d_model": 512,   # Example
        "encoder_layers": 4,      # Example
        "decoder_layers": 4,      # Example
        "num_heads": 8,           # Example
        "latent_dim": 128,        # Example (or 128)
        "element_padding_idx": 6,
        "command_padding_idx": 0,
        "num_bins": 256,           # Example
    }

try:
    checkpoint_vae = torch.load(PATH_VAE_CHECKPOINT, map_location=device) # Load to CPU first
    
    full_vae_model = model = VPVAE( # This is your hybrid VPVAE
        num_element_types=vp_vae_config["num_element_types"],
        num_command_types=vp_vae_config["num_command_types"],
        element_embed_dim=vp_vae_config["element_embed_dim"],
        command_embed_dim=vp_vae_config["command_embed_dim"],
        num_other_continuous_svg_features=vp_vae_config["num_other_continuous_svg_features"],
        num_other_continuous_params_to_reconstruct=vp_vae_config["num_other_continuous_svg_features"], # Decoder outputs same N continuous
        pixel_feature_dim=vp_vae_config["pixel_feature_dim"],
        encoder_d_model=vp_vae_config["encoder_d_model"], 
        decoder_d_model=vp_vae_config["decoder_d_model"],
        encoder_layers=vp_vae_config["encoder_layers"], 
        decoder_layers=vp_vae_config["decoder_layers"],
        num_heads=vp_vae_config["num_heads"], 
        latent_dim=vp_vae_config["latent_dim"], 
        max_seq_len=vp_vae_config["max_seq_len_train"],
        element_padding_idx=vp_vae_config["element_padding_idx"],
        command_padding_idx=vp_vae_config["command_padding_idx"]
    )
         
    full_vae_model.load_state_dict(checkpoint_vae); print("Full VAE weights loaded.")
    vae_decoder = full_vae_model.decoder.to(device).eval(); print("VAE Decoder extracted and set to eval mode.")
    del full_vae_model,  checkpoint_vae; gc.collect()
    
except Exception as e: print(f"Error loading VAE checkpoint: {e}"); exit()


# -----------------------------------------------------------------------------
# Load VS-DiT Model (Sequence Conditioning Version)
# -----------------------------------------------------------------------------
print("Instantiating VS-DiT model (Sequence Conditioning)...")
vs_dit_model = VS_DiT( # Use the correct imported class name
    latent_dim=VSDIT_LATENT_DIM,
    hidden_dim=VSDIT_HIDDEN_DIM,
    context_dim=VSDIT_CONTEXT_DIM, # Should match CLIP last_hidden_state
    num_blocks=VSDIT_NUM_BLOCKS,
    num_heads=VSDIT_NUM_HEADS,
    mlp_ratio=VSDIT_MLP_RATIO,
    dropout=VSDIT_DROPOUT
).to(device)

try:
    vs_dit_state_dict = torch.load(PATH_VSDIT_CHECKPOINT, map_location=device)
    vs_dit_model.load_state_dict(vs_dit_state_dict)
    vs_dit_model.eval()
    print(f"VS-DiT weights loaded successfully from '{PATH_VSDIT_CHECKPOINT}'.")
except FileNotFoundError: print(f"Error: VS-DiT checkpoint not found: {PATH_VSDIT_CHECKPOINT}"); exit()
except Exception as e: print(f"Error loading VS-DiT state dict: {e}"); exit()

# -----------------------------------------------------------------------------
# Load CLIP Model
# -----------------------------------------------------------------------------
print(f"Loading CLIP model from {PATH_CLIP_MODEL}...")
try:
    clip_model = CLIPTextModel.from_pretrained(PATH_CLIP_MODEL).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(PATH_CLIP_MODEL)
    clip_model.eval()
    # --- MODIFIED: Verify context dim matches LAST HIDDEN STATE size ---
    clip_output_dim = clip_model.config.hidden_size
    if VSDIT_CONTEXT_DIM != clip_output_dim:
         print(f"FATAL ERROR: VS-DiT context dim ({VSDIT_CONTEXT_DIM}) != CLIP last_hidden_state dim ({clip_output_dim}).")
         exit()
    print("CLIP model loaded successfully.")
except Exception as e: print(f"Error loading CLIP: {e}"); exit()

# -----------------------------------------------------------------------------
# Load Diffusion Parameters (Unchanged)
# -----------------------------------------------------------------------------
print("Setting up diffusion parameters...")
betas_cpu = get_linear_noise_schedule(DIFFUSION_NUM_TIMESTEPS)
diff_params = precompute_diffusion_parameters(betas_cpu, device)
print(f"Diffusion parameters precomputed for {DIFFUSION_NUM_TIMESTEPS} steps.")


# -----------------------------------------------------------------------------
# CLIP Embedding checking

def debug_clip_embeddings(prompt, device):
    with torch.no_grad():
        # Get conditional embedding
        text_inputs = clip_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(device)
        cond_emb = clip_model(**text_inputs).last_hidden_state
        
        # Get unconditional embedding
        empty_text = clip_tokenizer("", padding=True, truncation=True, return_tensors="pt").to(device)
        uncond_emb = clip_model(**empty_text).last_hidden_state
        
        print("\nCLIP Embedding Check:")
        print(f"Conditional shape: {cond_emb.shape}")
        print(f"Unconditional shape: {uncond_emb.shape}")
        print(f"Conditional: mean={cond_emb.mean():.4f}, std={cond_emb.std():.4f}")
        print(f"Unconditional: mean={uncond_emb.mean():.4f}, std={uncond_emb.std():.4f}")
        
        # Compare features dimension-wise
        print("\nFeature Statistics:")
        print(f"Conditional features mean: {cond_emb.mean(-1).mean():.4f}")
        print(f"Unconditional features mean: {uncond_emb.mean(-1).mean():.4f}")
        
        # Print attention masks
        cond_mask = text_inputs.attention_mask
        uncond_mask = empty_text.attention_mask
        print("\nMask Information:")
        print(f"Conditional mask shape: {cond_mask.shape}, sum: {cond_mask.sum().item()}")
        print(f"Unconditional mask shape: {uncond_mask.shape}, sum: {uncond_mask.sum().item()}")

        return cond_emb, uncond_emb

# -------------------------------------------------------------------------

def get_sub_schedule(diff_params, total_steps=1000, sampling_steps=100):
    indices = torch.linspace(0, total_steps - 1, sampling_steps).long()
    sub_params = {
        k: v[indices].clone() for k, v in diff_params.items()
        if v.shape[0] == total_steps
    }
    # Add previous alphas
    sub_params["alphas_cumprod_prev"] = torch.cat(
        [sub_params["alphas_cumprod"][:1], sub_params["alphas_cumprod"][:-1]], dim=0
    )
    return sub_params, indices


# -----------------------------------------------------------------------------
# Generation Function (MODIFIED)
# -----------------------------------------------------------------------------
def generate_svg_from_prompt(
    prompt: str,
    vs_dit_model: nn.Module,
    vae_decoder: nn.Module,
    clip_model: nn.Module,
    clip_tokenizer: object,
    diff_params: dict,
    device: torch.device,
    num_samples: int = 1,
    cfg_scale: float = 4.0,
    ddim_steps: int = 1000, # Using full steps for sampling
    ddim_eta: float = 0.0,
    num_diffusion_timesteps: int = 1000,
    vae_max_seq_len: int = 128,
    vae_num_bins: int = 256,
    vae_num_params: int = 10
):
    """Generates SVG tensor from text prompt using VS-DiT (SeqCond) and VAE Decoder."""
    print(f"\n--- Generating {num_samples} sample(s) for prompt: '{prompt}' ---")

    # 1. Get Text Embeddings (Last Hidden State) and Padding Mask
    text_inputs = clip_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_model.eval()

        debug_clip_embeddings(prompt, device)
        # --- MODIFIED: Get last_hidden_state and attention_mask ---
        text_outputs = clip_model(**text_inputs)
        context_sequence = text_outputs.last_hidden_state # Shape [1, S, context_dim]
        # Create key_padding_mask (True where padded) by inverting attention_mask
        context_padding_mask = ~(text_inputs.attention_mask.bool()) # Shape [1, S]

        # Repeat embeddings and mask if generating multiple samples
        if num_samples > 1:
             context_sequence = context_sequence.repeat(num_samples, 1, 1) # Shape [N, S, Ctx]
             context_padding_mask = context_padding_mask.repeat(num_samples, 1) # Shape [N, S]

    print(f"Text embedding seq shape: {context_sequence.shape}")
    print(f"Text padding mask shape: {context_padding_mask.shape}")

    # 2. Sample Latents using DDIM (SeqCond version)
    print(f"Sampling latents using DDIM ({ddim_steps} steps)...")
    latent_shape = (num_samples, VSDIT_LATENT_DIM)

    sampling_steps = ddim_steps
    sub_diff_params, step_indices = get_sub_schedule(diff_params, num_diffusion_timesteps, sampling_steps)

    # --- MODIFIED: Pass sequence and mask to ddim_sample ---
    sampled_latents = ddim_sample(
        model=vs_dit_model,
        shape=latent_shape,
        context_seq=context_sequence,             # Pass sequence
        context_padding_mask=context_padding_mask,# Pass mask
        diff_params=sub_diff_params,
        num_timesteps=sampling_steps,
        target_device=device,
        cfg_scale=cfg_scale,
        eta=ddim_eta,
        clip_tokenizer=clip_tokenizer,
        clip_model=clip_model
        # steps=ddim_steps # Add if ddim_sample supports variable steps
    )
    
    
    
    print(f"Sampled latents shape: {sampled_latents.shape} | Mean: {sampled_latents.mean():.4f}, Std: {sampled_latents.std():.4f}")

    # 3. Decode Latents using VAE Decoder (Unchanged)
    print("Decoding latents...")
    with torch.no_grad():
        vae_decoder.eval()
        element_logits, command_logits, param_logits_list = vae_decoder(sampled_latents, target_len=vae_max_seq_len)

    # 4. Process Decoder Output (Unchanged)
    print("Processing decoder output...")
    batch_size = sampled_latents.size(0)
    combined_outputs = []
    for i in range(batch_size):
        pred_elem_ids = element_logits[i].argmax(dim=-1)  
        #md_probs = F.softmax(command_logits[i], dim=-1)
        pred_cmds = command_logits[i].argmax(dim=-1)  
        pred_bin_indices_list = [
        torch.argmax(F.softmax(param_logits[i], dim=-1), dim=-1)  # [seq_len]
        for param_logits in param_logits_list
    ]
        pred_bin_indices = torch.stack(pred_bin_indices_list, dim=-1)  # [seq_len, num_params]
        # If you want [seq_len, num_params, 1]:
        pred_bin_indices = pred_bin_indices
        print(pred_elem_ids.long().unsqueeze(-1).shape, pred_cmds.long().unsqueeze(-1).shape, pred_bin_indices.shape)
    #pred_params_processed = torch.zeros_like(pred_bin_indices, dtype=torch.float32, device=device)
        # for param_idx in range(vae_num_params):
        #     param_type = PARAM_TYPES.get(param_idx, 'coord')
        #     bin_indices_for_param = pred_bin_indices[:, param_idx]
        #     if param_type == 'coord': pred_params_processed[:, param_idx] = dequantize_param_shifted(bin_indices_for_param, vae_num_bins, ORIGINAL_PARAM_MIN_COORD)
        #     elif param_type == 'rgb': pred_params_processed[:, param_idx] = dequantize_color_param(bin_indices_for_param, vae_num_bins)
        #     elif param_type == 'opacity': pred_params_processed[:, param_idx] = dequantize_opacity_param(bin_indices_for_param, vae_num_bins)
        #     elif param_type == 'strokewidth': pred_params_processed[:, param_idx] = dequantize_strokewidth_param(bin_indices_for_param, vae_num_bins, STROKEW_MAX)
        #     else: pred_params_processed[:, param_idx] = dequantize_param_shifted(bin_indices_for_param, vae_num_bins, ORIGINAL_PARAM_MIN_COORD) # Fallback
        combined_output = torch.cat([pred_elem_ids.long().unsqueeze(-1), pred_cmds.long().unsqueeze(-1), pred_bin_indices.long()], dim=-1)
        combined_outputs.append(combined_output)
    final_tensor_batch = torch.stack(combined_outputs, dim=0)

    return final_tensor_batch

# =============================================================================
# Main Execution Block (Unchanged logic, just uses modified function)
# =============================================================================
if __name__ == "__main__":
    prompts_to_generate = [ "purple umbrella with a grey handle"]
    n_samples_per_prompt = 5
    
    svg_tensor_converter_instance = SVGToTensor_Normalized()

    all_generated_tensors = []
    for prompt in prompts_to_generate:
        generated_tensors = generate_svg_from_prompt(
            prompt=prompt,
            vs_dit_model=vs_dit_model, vae_decoder=vae_decoder,
            clip_model=clip_model, clip_tokenizer=clip_tokenizer,
            diff_params=diff_params, device=device,
            num_samples=n_samples_per_prompt, cfg_scale=SAMPLING_CFG_SCALE,
            ddim_steps=SAMPLING_NUM_STEPS, ddim_eta=SAMPLING_ETA,
            num_diffusion_timesteps=DIFFUSION_NUM_TIMESTEPS,
            vae_max_seq_len=vp_vae_config["max_seq_len_train"], vae_num_bins=vp_vae_config["num_bins"],
            vae_num_params=vp_vae_config["num_other_continuous_svg_features"]
        )
        all_generated_tensors.append(generated_tensors)

    if all_generated_tensors:
        final_tensor_batch = torch.cat(all_generated_tensors, dim=0)
        print(f"\nFinal combined tensor shape: {final_tensor_batch.shape}")
        print("\nSaving generated SVGs...")
        # ... (SVG saving loop identical to previous script) ...
        prompt_idx = 0; sample_idx = 0
        for i in range(final_tensor_batch.size(0)):
            current_prompt = prompts_to_generate[prompt_idx]
            try:
                single_svg_tensor = final_tensor_batch[i]
                safe_prompt = current_prompt.replace(" ", "_").replace("/", "_")[:50]
                output_filename = f'generated_{safe_prompt}_s{sample_idx}.svg'
                output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                if single_svg_tensor.shape[0] != vp_vae_config["max_seq_len_train"]: print(f"Warning: Tensor length mismatch for {output_filename}")
                
                
                print(single_svg_tensor.shape)
                actual_recon_len = vp_vae_config["max_seq_len_train"]
                pred_elem_ids_list_cpu = single_svg_tensor[:,0].cpu().tolist()
                eos_id_int = int(svg_tensor_converter_instance.ELEMENT_TYPES['<EOS>'])
                pad_id_int = int(vp_vae_config["element_padding_idx"])
                for i_len_rec in range(single_svg_tensor.shape[0]):
                    if pred_elem_ids_list_cpu[i_len_rec] == eos_id_int or pred_elem_ids_list_cpu[i_len_rec] == pad_id_int:
                        actual_recon_len = i_len_rec 
                        break
                
                recon_svg = tensor_to_svg_file_hybrid_wrapper(single_svg_tensor, output_filename=output_filepath, 
                                                              svg_tensor_converter_instance=svg_tensor_converter_instance,
                                                              actual_len=actual_recon_len
                                                              )
                print(f"Saved: {output_filepath}")
                sample_idx += 1
                if sample_idx >= n_samples_per_prompt: sample_idx = 0; prompt_idx += 1
            except Exception as e:
                print(f"Error saving SVG for '{current_prompt}', sample {sample_idx}: {str(e)}")
                sample_idx += 1 # Ensure loop progresses even on error
                if sample_idx >= n_samples_per_prompt: sample_idx = 0; prompt_idx += 1
    else: print("No tensors were generated.")

# --- END OF FILE generate_samples.py ---