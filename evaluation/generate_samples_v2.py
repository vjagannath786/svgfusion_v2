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
    from models import VS_DiT, get_linear_noise_schedule, precompute_diffusion_parameters, ddim_sample, ddim_sample_fixed
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
PATH_VAE_CHECKPOINT = './best_models/vp_vae_accel_hybrid_fallen-jazz-36_s13500_best.pt' # VAE trained with sequential latents
PATH_VSDIT_CHECKPOINT = "./saved_models_vsdit_clip_second_patch/vsdit_clip_seqcond_misty-valley-113_best.pth" # DiT trained with sequential latents
PATH_CLIP_MODEL = "/Users/varun_jagannath/Documents/D/test python/clip-vit-large-patch14" # HuggingFace model name or local path
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
SAMPLING_CFG_SCALE = 1.5 # Classifier-Free Guidance scale
SAMPLING_ETA = 0.0 # 0.0 for deterministic DDIM

# --- Setup Device ---
if torch.cuda.is_available(): device = torch.device("cuda") # Use cuda if available, otherwise cpu
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(42) # Set seed for reproducibility


# In generate_samples_v1.py, near the top configuration section
PATH_ZDATASET = "./zdataset_vpvae_patch_tokens.pt" # Path to the dataset object

# Load the dataset to get normalization stats
print(f"Loading zDataset from {PATH_ZDATASET} to get normalization stats...")
try:
    # Load zDataset class definition first
    from models import zDataset
    full_zdataset = torch.load(PATH_ZDATASET)
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
    checkpoint_vae = torch.load(PATH_VAE_CHECKPOINT, map_location='cpu') # Load to CPU first
    
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
    vae_num_params_svg: int = VAE_NUM_OTHER_CONT_FEATURES # From VAE config
):
    print(f"\n--- Generating {num_samples} sample(s) for prompt: '{prompt}' ---")

    # 1. Get Text Embeddings for Conditional Context
    text_inputs = clip_tokenizer([prompt] * num_samples, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_model.eval()
        text_outputs = clip_model(**text_inputs)
        context_sequence = text_outputs.last_hidden_state # [num_samples, S_text, D_clip]
        context_padding_mask = ~(text_inputs.attention_mask.bool()) # [num_samples, S_text]

    # 2. Define Latent Shape for SAMPLING
    latent_shape = (num_samples, latent_seq_len, latent_feature_dim)
    print(f"Target latent shape for DiT sampling: {latent_shape}")

    # 2.A. Sample Latents using DDIM
    print(f"Sampling latents using DDIM ({ddim_steps} steps)...")
    sampled_latents, _ = ddim_sample_fixed( # ddim_sample_improved returns (z_t, stats_dict)
        model=vs_dit_model,
        shape=latent_shape,
        context_seq=context_sequence,
        context_padding_mask=context_padding_mask,
        diff_params=ddim_diff_params,
        num_timesteps=ddim_steps,
        target_device=device,
        cfg_scale=cfg_scale,
        eta=ddim_eta,
        clip_tokenizer=clip_tokenizer, # Passed for internal unconditional context generation
        clip_model=clip_model,       # Passed for internal unconditional context generation
        return_visuals=False # No plots during generation loop by default
    )
    # sampled_latents: [num_samples, LATENT_SEQ_LEN, LATENT_FEATURE_DIM]
    
    print(f"Sampled latents shape: {sampled_latents.shape} | Mean: {sampled_latents.mean():.4f}, Std: {sampled_latents.std():.4f}")
    
    # --- ADD THIS UN-NORMALIZATION STEP ---
    unnormalized_latents = sampled_latents * z_std + z_mean
    print(f"Un-normalized latents shape: {unnormalized_latents.shape} | Mean: {unnormalized_latents.mean():.4f}, Std: {unnormalized_latents.std():.4f}")
    
    #vae_latent_scale_factor = 0.4 
    #scaled_latents = unnormalized_latents * vae_latent_scale_factor
    
    #print(f"Final scaled latents shape: {scaled_latents.shape} | Mean: {scaled_latents.mean():.4f}, Std: {scaled_latents.std():.4f}")

    # 3. Decode Latents using VAE Decoder
    print("Decoding latents...")
    with torch.no_grad():
        vae_decoder.eval()
        element_logits, command_logits, param_logits_list = vae_decoder(
            unnormalized_latents,
            target_len=vae_target_svg_len # Length of the SVG command sequence to generate
        )

    # 4. Process Decoder Output
    print("Processing decoder output...")
    batch_size_decode = unnormalized_latents.size(0)
    combined_outputs = []
    
    # --- SVGToTensor_Normalized (for getting PAD/EOS IDs consistently) ---
    svg_tensor_converter_decoder_ref = SVGToTensor_Normalized()
    _element_padding_idx = svg_tensor_converter_decoder_ref.ELEMENT_TYPES.get('<PAD>', 0) 
    _eos_id_int = svg_tensor_converter_decoder_ref.ELEMENT_TYPES.get('<EOS>', 0)


    for i in range(batch_size_decode):
        pred_elem_ids = element_logits[i].argmax(dim=-1)
        pred_cmds = command_logits[i].argmax(dim=-1)
        pred_bin_indices_list = [
            torch.argmax(F.softmax(param_logits[i], dim=-1), dim=-1) # Use softmax for proper argmax
            for param_logits in param_logits_list
        ]
        pred_bin_indices = torch.stack(pred_bin_indices_list, dim=-1)
        
        combined_output = torch.cat([
            pred_elem_ids.long().unsqueeze(-1), 
            pred_cmds.long().unsqueeze(-1), 
            pred_bin_indices.long()
        ], dim=-1) # combined_output: [VAE_MAX_SEQ_LEN_SVG_CMDS, num_svg_params_total]
        combined_outputs.append(combined_output)
    final_tensor_batch = torch.stack(combined_outputs, dim=0) # [B, VAE_MAX_SEQ_LEN_SVG_CMDS, num_svg_params_total]


    return final_tensor_batch

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    prompts_to_generate = [
    #    "a cute smiling cat",
    #     "a red umbrella",
    #     "a futuristic robot",
    #     "a simple green tree",
    #     "a blue abstract shape",
        "a circle"
    ]
    n_samples_per_prompt = 2 # Generate 2 samples for each prompt

    svg_tensor_converter_instance = SVGToTensor_Normalized() # For final SVG reconstruction
    
    # --- Get PAD/EOS IDs from a consistent source (SVGToTensor_Normalized) ---
    # These are needed for actual_len detection in tensor_to_svg_file_hybrid_wrapper
    _element_padding_idx_for_final_save = svg_tensor_converter_instance.ELEMENT_TYPES.get('<PAD>', 0)
    _eos_id_int_for_final_save = svg_tensor_converter_instance.ELEMENT_TYPES.get('<EOS>', 0)

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
            vae_num_params_svg=VAE_NUM_OTHER_CONT_FEATURES
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
