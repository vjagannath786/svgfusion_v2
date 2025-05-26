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
import sys
import traceback
sys.path.append('/home/svgfusion_v2/') # Assuming models and svgutils are in this directory or subdirs

# --- Import VAE Components ---
try:
    # Ensure VPVAE is the version that outputs/accepts sequential latents
    from models import (VPVAE, set_seed)
    print("Successfully imported VAE components.")
except ImportError as e:
    print(f"Error importing VAE components: {e}"); exit()

# --- Import VS-DiT Model and Diffusion Utils (Sequence Conditioning Version) ---
try:
    # Ensure VS_DiT is the version that processes sequential latents
    # and that ddim_sample/dpm_solver handle these shapes
    from models import VS_DiT
    from models import get_linear_noise_schedule, precompute_diffusion_parameters, ddim_sample, dpm_solver_2m_20_steps, ddim_sample_improved
    print("Successfully imported VS-DiT (SeqCond) model and diffusion utilities.")
except ImportError as e:
    print(f"Error importing DiT model/utils: {e}"); exit()
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
PATH_VAE_CHECKPOINT = '/home/model_weights/vp_vae_accel_hybrid_light-feather-36_s5000_best.pt' # VAE trained with sequential latents
PATH_VSDIT_CHECKPOINT = "/home/saved_models_vsdit_clip/vsdit_clip_seqcond_fresh-haze-66_best.pth" # DiT trained with sequential latents
PATH_CLIP_MODEL = "/home/clip_model/clip-vit-large-patch14"
OUTPUT_DIR = "./generated_svgs_vsdit_seqcond_sequential_latent" # New output dir

# --- Model & Sampling Parameters ---
# These MUST match the parameters used for training the loaded VS_DiT checkpoint
# For a DiT processing SEQUENTIAL latents:
LATENT_SEQ_LEN = 256  # NEW: Sequence length of the VAE latent (e.g., after pooling)
LATENT_FEATURE_DIM = 128 # NEW: Feature dimension of each token in the latent sequence
                         # This is likely what VSDIT_LATENT_DIM previously referred to if DiT projects it.
                         # Or, if DiT's hidden_dim directly processes this, VSDIT_LATENT_DIM might be hidden_dim.
                         # Let's assume DiT's input_dim_per_token is LATENT_FEATURE_DIM

VSDIT_INPUT_DIM_PER_TOKEN = LATENT_FEATURE_DIM # Explicitly state what DiT's proj_in expects per token
# VSDIT_LATENT_DIM = 128 # This might now be redundant if we use the above two

VSDIT_HIDDEN_DIM = 384
VSDIT_CONTEXT_DIM = 768
VSDIT_NUM_BLOCKS = 12
VSDIT_NUM_HEADS = 6
VSDIT_MLP_RATIO = 8.0 # Or 8.0 if that was your S-model config
VSDIT_DROPOUT = 0.0

DIFFUSION_NUM_TIMESTEPS = 1000
SAMPLING_NUM_STEPS_DDIM = 100 # Fewer steps for DDIM if preferred, or 1000 for quality
SAMPLING_NUM_STEPS_DPM = 20   # For DPM-Solver
SAMPLING_CFG_SCALE = 3
SAMPLING_ETA = 0.0

# --- VAE parameters (should match the loaded VAE) ---
# These are mostly for the decoder's output target_len and processing SVG tensors
VAE_MAX_SEQ_LEN_SVG_CMDS = 1024 # Max sequence length of the *SVG command sequence*
VAE_LATENT_DIM_PER_TOKEN_FROM_VAE = LATENT_FEATURE_DIM # Feature dim of each token in the VAE's sequential latent output
VAE_LATENT_SEQ_LEN_FROM_VAE = LATENT_SEQ_LEN         # Sequence length of VAE's sequential latent output

# --- Dequantization Constants & SVG processing ---
# ... (these remain the same) ...
PARAM_TYPES = { 0: 'coord', 1: 'coord', 2: 'coord', 3: 'coord', 4: 'rgb', 5: 'rgb', 6: 'rgb', 7: 'opacity', 8: 'rgb', 9: 'rgb'}


# --- Setup Device ---
if torch.cuda.is_available(): device = torch.device("cuda:1")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Load VAE (Ensure it's the version outputting sequential latents)
# -----------------------------------------------------------------------------
print(f"Loading VAE checkpoint from '{PATH_VAE_CHECKPOINT}'...")
# The vp_vae_config should reflect the VAE that produces sequential latents
# Specifically, its `latent_dim` should be `LATENT_FEATURE_DIM` (e.g., 128)
# and its encoder should output [B, LATENT_SEQ_LEN, LATENT_FEATURE_DIM]
# and its decoder should accept this shape.
vp_vae_config = {
    "max_seq_len_train": VAE_MAX_SEQ_LEN_SVG_CMDS, # For SVG command sequence length
    "num_element_types": 7, # Example, use actual
    "num_command_types": 14, # Example, use actual
    "element_embed_dim": 64,
    "command_embed_dim": 64,
    "num_other_continuous_svg_features": 12, # Example
    "pixel_feature_dim": 384, # Example
    "encoder_d_model": 512, # Example, from your VAE training script
    "decoder_d_model": 512, # Example
    "encoder_layers": 4,
    "decoder_layers": 4,
    "num_heads": 8,
    "latent_dim": LATENT_FEATURE_DIM, # Crucial: Feature dim per token of the VAE's *sequential* latent
    "element_padding_idx": 6, # Example
    "command_padding_idx": 0, # Example
    "num_bins": 256,
}

try:
    checkpoint_vae = torch.load(PATH_VAE_CHECKPOINT, map_location=device)
    full_vae_model = VPVAE( num_element_types=vp_vae_config["num_element_types"],
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
        command_padding_idx=vp_vae_config["command_padding_idx"]) # Pass the config
    full_vae_model.load_state_dict(checkpoint_vae)
    print("Full VAE (sequential latent version) weights loaded.")
    vae_decoder = full_vae_model.decoder.to(device).eval()
    print("VAE Decoder (sequential latent version) extracted and set to eval mode.")
    del full_vae_model, checkpoint_vae; gc.collect()
except Exception as e: print(f"Error loading VAE checkpoint: {e}"); traceback.print_exc(); exit()

# -----------------------------------------------------------------------------
# Load VS-DiT Model (Ensure it's version for sequential latents)
# -----------------------------------------------------------------------------
print("Instantiating VS-DiT model (for Sequential Latents)...")
# The DiT's `latent_dim` param should be VSDIT_INPUT_DIM_PER_TOKEN
# And it needs `input_sequence_length_N` if it uses fixed positional embeddings.
# Assuming your VS_DiT is modified like:
# def __init__(self, input_dim_per_token, sequence_length_N, hidden_dim, context_dim_clip, ...)
# For simplicity, let's assume your VS_DiT's `latent_dim` argument now means `input_dim_per_token`
# and it internally knows its expected `sequence_length_N` (e.g., via a new arg or fixed).
# If VS_DiT's `latent_dim` still refers to the input feature dim per token, that's VSDIT_INPUT_DIM_PER_TOKEN
vs_dit_model = VS_DiT(
    latent_dim=VSDIT_INPUT_DIM_PER_TOKEN, # This should be feature_dim of each token in z_t
    # sequence_length_N=LATENT_SEQ_LEN, # Add if your DiT needs it for pos_embed
    hidden_dim=VSDIT_HIDDEN_DIM,
    context_dim=VSDIT_CONTEXT_DIM,
    num_blocks=VSDIT_NUM_BLOCKS,
    num_heads=VSDIT_NUM_HEADS,
    mlp_ratio=VSDIT_MLP_RATIO,
    dropout=VSDIT_DROPOUT
).to(device)

try:
    vs_dit_state_dict = torch.load(PATH_VSDIT_CHECKPOINT, map_location=device)
    vs_dit_model.load_state_dict(vs_dit_state_dict)
    vs_dit_model.eval()
    print(f"VS-DiT (sequential latent version) weights loaded from '{PATH_VSDIT_CHECKPOINT}'.")
except Exception as e: print(f"Error loading VS-DiT state dict: {e}"); traceback.print_exc(); exit()

# -----------------------------------------------------------------------------
# Load CLIP Model (Unchanged)
# -----------------------------------------------------------------------------
# ... (CLIP loading code remains the same) ...
print(f"Loading CLIP model from {PATH_CLIP_MODEL}...")
try:
    clip_model = CLIPTextModel.from_pretrained(PATH_CLIP_MODEL).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(PATH_CLIP_MODEL)
    clip_model.eval()
    clip_output_dim = clip_model.config.hidden_size
    if VSDIT_CONTEXT_DIM != clip_output_dim:
         print(f"FATAL ERROR: VS-DiT context dim ({VSDIT_CONTEXT_DIM}) != CLIP last_hidden_state dim ({clip_output_dim}).")
         exit()
    print("CLIP model loaded successfully.")
except Exception as e: print(f"Error loading CLIP: {e}"); traceback.print_exc(); exit()

# -----------------------------------------------------------------------------
# Load Diffusion Parameters (Unchanged, relates to timesteps T)
# -----------------------------------------------------------------------------
# ... (Diffusion params code remains the same) ...
print("Setting up diffusion parameters...")
betas_cpu = get_linear_noise_schedule(DIFFUSION_NUM_TIMESTEPS)
diff_params = precompute_diffusion_parameters(betas_cpu, device) # For DDIM
print(f"Diffusion parameters precomputed for {DIFFUSION_NUM_TIMESTEPS} steps.")


# -----------------------------------------------------------------------------
# Generation Function
# -----------------------------------------------------------------------------
def generate_svg_from_prompt(
    prompt: str,
    vs_dit_model: nn.Module,
    vae_decoder: nn.Module, # This decoder expects [B, LATENT_SEQ_LEN, LATENT_FEATURE_DIM]
    clip_model: nn.Module,
    clip_tokenizer: object,
    device: torch.device,
    num_samples: int = 1,
    cfg_scale: float = 3.0,
    # DDIM specific
    ddim_diff_params: dict = None, # Pass the precomputed diff_params for DDIM
    ddim_steps: int = 1000,
    ddim_eta: float = 0.0,
    # DPM-Solver specific
    dpm_betas: torch.Tensor = None, # Pass betas for DPM solver
    dpm_total_train_timesteps: int = 1000,
    # General
    latent_seq_len: int = LATENT_SEQ_LEN,
    latent_feature_dim: int = LATENT_FEATURE_DIM,
    vae_target_svg_len: int = VAE_MAX_SEQ_LEN_SVG_CMDS, # For decoder's target_len argument
    # ... other vae params for decoding SVG (num_bins, num_params)
    vae_num_params_svg: int = 12 # From your vp_vae_config
):
    print(f"\n--- Generating {num_samples} sample(s) for prompt: '{prompt}' ---")

    # 1. Get Text Embeddings (Unchanged conceptually)
    # ... (debug_clip_embeddings and context_sequence generation as before) ...
    text_inputs = clip_tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_model.eval()
        text_outputs = clip_model(**text_inputs)
        context_sequence = text_outputs.last_hidden_state
        context_padding_mask = ~(text_inputs.attention_mask.bool())
        if num_samples > 1:
             context_sequence = context_sequence.repeat(num_samples, 1, 1)
             context_padding_mask = context_padding_mask.repeat(num_samples, 1)

    # 2. Define Latent Shape for SAMPLING
    # THIS IS THE KEY CHANGE: latents are now sequences
    latent_shape = (num_samples, latent_seq_len, latent_feature_dim)
    print(f"Target latent shape for DiT sampling: {latent_shape}")

    # 2.A. Sample Latents using DDIM
    print(f"Sampling latents using DDIM ({ddim_steps} steps)...")
    # sub_diff_params, _ = get_sub_schedule(ddim_diff_params, dpm_total_train_timesteps, ddim_steps) # dpm_total_train_timesteps is just T
    sampled_latents_ddim  = ddim_sample_improved( # Assuming ddim_sample is updated for sequential latents
        model=vs_dit_model,
        shape=latent_shape,
        context_seq=context_sequence,
        context_padding_mask=context_padding_mask,
        diff_params=ddim_diff_params, # Use full diff_params if ddim_sample handles sub_scheduling, or pass sub_diff_params
        num_timesteps=ddim_steps, # Or dpm_total_train_timesteps if ddim_sample expects T
        target_device=device,
        cfg_scale=cfg_scale,
        eta=ddim_eta,
        clip_tokenizer=clip_tokenizer,
        clip_model=clip_model,
        return_visuals=True # No plots during generation loop by default
    )
    sampled_latents = sampled_latents_ddim # Choose which sampler's output to use

    # 2.B. Sample Latents using DPM-Solver
    # print(f"Sampling latents using DPM-Solver ({SAMPLING_NUM_STEPS_DPM} steps)...")
    # sampled_latents_dpm = dpm_solver_2m_20_steps( # Ensure this solver handles sequential latents
    #     model=vs_dit_model,
    #     shape=latent_shape,
    #     prompt=prompt, # DPM solver helper might re-tokenize, or pass embeddings
    #     uncond_prompt="",
    #     clip_model=clip_model,
    #     clip_tokenizer=clip_tokenizer,
    #     total_train_timesteps=dpm_total_train_timesteps,
    #     betas=dpm_betas, # Pass the full beta schedule
    #     cfg_scale=cfg_scale,
    #     device=device,
    #     seed=42 # Give a random seed per call or fix
    # )
    # sampled_latents = sampled_latents_dpm # Choose DPM solver output


    print(f"Sampled latents shape: {sampled_latents.shape} | Mean: {sampled_latents.mean():.4f}, Std: {sampled_latents.std():.4f}")

    # Optional: Rescale latents if their stats differ too much from VAE training
    # mean_sl = sampled_latents.mean(dim=(-2, -1), keepdim=True) # Mean over seq_len and features
    # std_sl = sampled_latents.std(dim=(-2, -1), keepdim=True)   # Std over seq_len and features
    # target_std_vae = 1.0 # Or actual std of VAE training latents (if normalized)
    # sampled_latents = (sampled_latents - mean_sl) / (std_sl + 1e-7) * target_std_vae
    # print(f"Rescaled latents stats: Mean: {sampled_latents.mean():.4f}, Std: {sampled_latents.std():.4f}")


    # 3. Decode Latents using VAE Decoder
    # The VAE decoder now expects z of shape [B, LATENT_SEQ_LEN, LATENT_FEATURE_DIM]
    # Its `target_len` argument refers to the output SVG command sequence length
    print("Decoding latents...")
    with torch.no_grad():
        vae_decoder.eval()
        element_logits, command_logits, param_logits_list = vae_decoder(
            sampled_latents,
            target_len=vae_target_svg_len # Length of the SVG command sequence to generate
        )

    # 4. Process Decoder Output (Mostly Unchanged, operates on token predictions)
    # ... (SVG processing logic remains largely the same as it works per token) ...
    print("Processing decoder output...")
    batch_size_decode = sampled_latents.size(0) # Use actual batch size from latents
    combined_outputs = []
    for i in range(batch_size_decode):
        pred_elem_ids = element_logits[i].argmax(dim=-1)
        pred_cmds = command_logits[i].argmax(dim=-1)
        pred_bin_indices_list = [
            torch.argmax(F.softmax(param_logits[i], dim=-1), dim=-1)
            for param_logits in param_logits_list
        ]
        pred_bin_indices = torch.stack(pred_bin_indices_list, dim=-1)
        # print(f"Debug shapes in SVG processing: elem_ids: {pred_elem_ids.long().unsqueeze(-1).shape}, cmds: {pred_cmds.long().unsqueeze(-1).shape}, bins: {pred_bin_indices.shape}")
        combined_output = torch.cat([pred_elem_ids.long().unsqueeze(-1), pred_cmds.long().unsqueeze(-1), pred_bin_indices.long()], dim=-1)
        combined_outputs.append(combined_output)
    final_tensor_batch = torch.stack(combined_outputs, dim=0)


    return final_tensor_batch

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    prompts_to_generate = [ "a red circle"] # "a smiling emoji face", "an arrow pointing right"
    n_samples_per_prompt = 5 # For quicker testing

    svg_tensor_converter_instance = SVGToTensor_Normalized() # For SVG final processing

    all_generated_tensors = []
    for prompt_text in prompts_to_generate: # Renamed variable
        generated_tensors = generate_svg_from_prompt(
            prompt=prompt_text,
            vs_dit_model=vs_dit_model,
            vae_decoder=vae_decoder,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            device=device,
            num_samples=n_samples_per_prompt,
            cfg_scale=SAMPLING_CFG_SCALE,
            # DDIM
            ddim_diff_params=diff_params, # Full diff_params for DDIM
            ddim_steps=SAMPLING_NUM_STEPS_DDIM,
            ddim_eta=SAMPLING_ETA,
            # DPM
            dpm_betas=betas_cpu, # Original CPU betas for DPM solver
            dpm_total_train_timesteps=DIFFUSION_NUM_TIMESTEPS,
            # Latent structure
            latent_seq_len=LATENT_SEQ_LEN,
            latent_feature_dim=LATENT_FEATURE_DIM,
            # VAE
            vae_target_svg_len=VAE_MAX_SEQ_LEN_SVG_CMDS,
            vae_num_params_svg=vp_vae_config["num_other_continuous_svg_features"]
        )
        all_generated_tensors.append(generated_tensors)

    if all_generated_tensors:
        final_tensor_batch = torch.cat(all_generated_tensors, dim=0)
        print(f"\nFinal combined tensor shape: {final_tensor_batch.shape}") # Should be [total_samples, VAE_MAX_SEQ_LEN_SVG_CMDS, num_svg_params_total]
        print("\nSaving generated SVGs...")
        
        prompt_idx_save = 0 # Corrected variable name
        sample_idx_save = 0 # Corrected variable name
        for i in range(final_tensor_batch.size(0)):
            current_prompt = prompts_to_generate[prompt_idx_save]
            try:
                single_svg_tensor = final_tensor_batch[i]
                safe_prompt_filename = current_prompt.replace(" ", "_").replace("/", "_")[:50] # Corrected var name
                output_filename = f'generated_{safe_prompt_filename}_s{sample_idx_save}.svg'
                output_filepath = os.path.join(OUTPUT_DIR, output_filename)
                
                # Determine actual length (before padding or EOS) for reconstruction
                actual_recon_len = VAE_MAX_SEQ_LEN_SVG_CMDS # Start with max
                pred_elem_ids_list_cpu = single_svg_tensor[:,0].cpu().tolist()
                eos_id_int = int(svg_tensor_converter_instance.ELEMENT_TYPES['<EOS>'])
                pad_id_int = int(vp_vae_config["element_padding_idx"]) # Use pad_idx from config
                for i_len_rec in range(single_svg_tensor.shape[0]):
                    if pred_elem_ids_list_cpu[i_len_rec] == eos_id_int or \
                       pred_elem_ids_list_cpu[i_len_rec] == pad_id_int:
                        actual_recon_len = i_len_rec
                        break
                
                print(f"Reconstructing SVG with actual_len={actual_recon_len} for {output_filename}")
                recon_svg = tensor_to_svg_file_hybrid_wrapper(
                    single_svg_tensor, # Should be [VAE_MAX_SEQ_LEN_SVG_CMDS, num_params_total]
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