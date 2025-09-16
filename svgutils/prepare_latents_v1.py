import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys
import traceback
import json
from pathlib import Path

sys.path.append(".")

try:
    from models import VPVAE, set_seed
    from svgutils import SVGToTensor_Normalized, load_dino_model_components
except ImportError as e:
    print(f"CRITICAL ERROR importing VPVAE or SVGToTensor_Normalized: {e}")
    sys.exit(1)
    
# --- Paths and Config ---
PRECOMPUTED_DATA_OUTPUT_DIR = "precomputed_patch_tokens_data" 
PRECOMPUTED_FILE_LIST_PATH = "precomputed_patch_tokens_file_list.pt" # File containing list of paths

MODEL_PATH = './best_models/vp_vae_accel_hybrid_olive-darkness-31_s9500_best.pt' 
OUTPUT_Z_DATASET_PATH = 'zdataset_vpvae_patch_tokens.pt' # Output file for the zDataset

with open('./svg_captions.json', 'r') as f:
    svg_captions = json.load(f)
    

# --- zDataset Definition ---
class zDataset(Dataset):
    def __init__(self, z_data_list):
        # z_data_list is a list of dicts: [{'filename': str, 'z': np.array, 'text': str}, ...]
        self.z_data = z_data_list
        # Stack z tensors for statistics calculation, ensure they are torch.Tensors
        # Assuming z is already [L_svg, latent_dim] numpy array
        all_z_tensors = [torch.from_numpy(item['z']) for item in z_data_list]
        
        # We need to handle variable sequence lengths of z
        # For overall stats, we can concatenate all valid z tokens
        concatenated_valid_z = []
        for z_tensor in all_z_tensors:
            # Assuming padding tokens in z are zeros and we want to exclude them
            # This needs to be robust to how padding is handled in `z` in VPVAEEncoder
            # For simplicity, if `z` for padded tokens is zeroed out, we can filter.
            # A better way would be to pass the actual svg_mask_batch from preprocessing.
            # For now, let's just use all provided z, or average/pool if one overall z is desired.
            # If z is [L_svg, latent_dim], and L_svg can be variable per original SVG:
            concatenated_valid_z.append(z_tensor)

        if concatenated_valid_z:
            # If each z_tensor is [L_svg, latent_dim], stack to [Num_SVGs, L_svg_max, latent_dim]
            # or concatenate all valid tokens to get a single long sequence for global stats.
            # Let's compute stats per token for a specific length if possible, or overall mean/std
            # For global stats across *all valid tokens*:
            all_valid_z_tokens_flat = torch.cat([z_t.view(-1, z_t.shape[-1]) for z_t in concatenated_valid_z if z_t.numel() > 0], dim=0)
            if all_valid_z_tokens_flat.numel() > 0:
                self.z_mean = all_valid_z_tokens_flat.mean(0, keepdim=True) # [1, latent_dim]
                self.z_std = all_valid_z_tokens_flat.std(0, keepdim=True)   # [1, latent_dim]
            else:
                self.z_mean = torch.zeros(1, z_data_list[0]['z'].shape[-1])
                self.z_std = torch.ones(1, z_data_list[0]['z'].shape[-1])
        else:
            # Fallback if z_data_list is empty
            self.z_mean = torch.zeros(1, z_data_list[0]['z'].shape[-1] if z_data_list else 32) # Default latent_dim
            self.z_std = torch.ones(1, z_data_list[0]['z'].shape[-1] if z_data_list else 32)
        
        print(f"zDataset initialized. Global Z mean: {self.z_mean.mean().item():.4f}, std: {self.z_std.mean().item():.4f}")

    def __len__(self):
        return len(self.z_data)

    def __getitem__(self, idx):
        item = self.z_data[idx]
        # Ensure z is converted to torch.Tensor when retrieved
        return {'z': torch.from_numpy(item['z']), 'text' : item['text'], 'filename': item['filename']}
    


if __name__ == "__main__":
    set_seed(42)
    
    device = torch.device("cpu") # Changed to CPU for data loading/preprocessing
    print(f"Using device for data preparation: {device}")
    
    # --- Dynamically Infer Configuration Parameters ---
    print("Inferring model configuration from data and DINOv2 model...")
    
    dino_model_dims, dino_processor_dims, _, dino_embed_dim_from_data, fixed_dino_patch_seq_length = load_dino_model_components()
    
    del dino_model_dims, dino_processor_dims
    
    try:
        precomputed_file_paths = torch.load(PRECOMPUTED_FILE_LIST_PATH, map_location='cpu')
        if not precomputed_file_paths: 
            print("Error: Precomputed file list is empty. Cannot infer SVG dimensions."); sys.exit(1)
    except Exception as e:
        print(f"Error loading precomputed file paths from {PRECOMPUTED_FILE_LIST_PATH}: {e}"); traceback.print_exc(); sys.exit(1)
        
    
    try:
        first_precomputed_item_data = torch.load(precomputed_file_paths[0], map_location='cpu')
        num_total_svg_features_from_data = first_precomputed_item_data['full_svg_matrix_content'].shape[1]
        num_other_continuous_features_cfg = num_total_svg_features_from_data - 2 
    except Exception as e:
        print(f"Error loading first precomputed item to infer SVG dimensions: {e}"); traceback.print_exc(); sys.exit(1)
    
    temp_svg_tensor_converter_for_config = SVGToTensor_Normalized() 
    num_element_types_cfg = len(temp_svg_tensor_converter_for_config.ELEMENT_TYPES)
    num_command_types_cfg = len(temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES)
    
    element_pad_idx_cfg = temp_svg_tensor_converter_for_config.ELEMENT_TYPES.get('<PAD>', 0) 
    command_pad_idx_cfg = temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES.get('NO_CMD', 0)
    
    # --- Model Configuration (MUST MATCH THE TRAINED MODEL'S CONFIG!) ---
    # Update these values with the EXACT configuration from your trained model's wandb run or config file.
    # The dynamically inferred values are used to *verify and set* these config items.
    config = {
        "max_seq_len_train": 1024,
        "num_element_types": num_element_types_cfg,
        "num_command_types": num_command_types_cfg,
        "element_embed_dim": 64,  
        "command_embed_dim": 64,  
        "num_other_continuous_svg_features": num_other_continuous_features_cfg,
        "pixel_feature_dim": dino_embed_dim_from_data, 
        "encoder_d_model": 512,   
        "decoder_d_model": 512,   
        "encoder_layers": 4,      
        "decoder_layers": 4,      
        "num_heads": 8,           
        "latent_dim": 32, 
        "element_padding_idx": element_pad_idx_cfg,
        "command_padding_idx": command_pad_idx_cfg,
        "fixed_dino_patch_seq_length": fixed_dino_patch_seq_length
    }
    print(f"Using inferred and confirmed model config: {config}")
    
    # --- Instantiate VPVAE Model ---
    print("Instantiating VPVAE model...")
    model = VPVAE(
        num_element_types=config["num_element_types"],
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
        command_padding_idx=config["command_padding_idx"]
    )
    
    # --- Load Model State Dictionary ---
    print(f"Loading model state_dict from '{MODEL_PATH}'...")
    try:
        # Load state_dict to CPU first, then move model to desired device
        loaded_state_dict = torch.load(MODEL_PATH, map_location='cpu') 
        model.load_state_dict(loaded_state_dict)
        model.to(device) # Move model to device after loading state_dict
        print("Model weights applied and model moved to device successfully.")
    except Exception as e: 
        print(f"Error loading or applying state_dict: {e}"); traceback.print_exc(); sys.exit(1)
    model.eval()
    
    # --- Construct SOS/EOS/PAD tokens ---
    # These must use the dynamically inferred num_other_continuous_features_cfg
    default_cont_param_values_for_tokens = torch.full(
        (config["num_other_continuous_svg_features"],), 
        temp_svg_tensor_converter_for_config.DEFAULT_PARAM_VAL, dtype=torch.float32
    )
    sos_elem_id_val = float(temp_svg_tensor_converter_for_config.ELEMENT_TYPES['<BOS>'])
    eos_elem_id_val = float(temp_svg_tensor_converter_for_config.ELEMENT_TYPES['<EOS>'])
    pad_elem_id_val = float(config["element_padding_idx"]) 
    no_cmd_id_val = float(config["command_padding_idx"])   
    
    sos_token_row_input = torch.cat([torch.tensor([sos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])
    eos_token_row_input = torch.cat([torch.tensor([eos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])
    padding_row_template_input = torch.cat([torch.tensor([pad_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])

    z_data_list = []
    
    print(f"\nEncoding {len(precomputed_file_paths)} SVGs into latent space...")
    with torch.no_grad():
        for file_path in tqdm(precomputed_file_paths, desc="Preparing Latents"):
            try:
                # --- Load item_data on demand ---
                item_data = torch.load(file_path, map_location='cpu') 
                filename_stem = item_data['file_name_stem']
                
                full_svg_content_unpadded = item_data['full_svg_matrix_content'].to(device)
                
                # --- Get final pixel patch tokens sequence (Keys/Values for encoder) ---
                # Taking the last stage's cumulative pixel patch tokens
                final_pixel_patch_tokens_seq = item_data['cumulative_pixel_patch_tokens_aligned'][-1, :, :].to(device) # Shape: [M, D_dino]

                # --- Prepare ENCODER input for SVG (Queries) ---
                svg_enc_input_unp = torch.cat([
                    sos_token_row_input.unsqueeze(0).to(device), full_svg_content_unpadded, eos_token_row_input.unsqueeze(0).to(device)
                ], dim=0)
                len_svg_unp = svg_enc_input_unp.shape[0]
                
                svg_enc_input_pad = svg_enc_input_unp.clone()
                svg_attn_mask = torch.zeros(len_svg_unp, dtype=torch.bool, device=device) # Query mask

                if len_svg_unp > config["max_seq_len_train"]:
                    svg_enc_input_pad = svg_enc_input_unp[:config["max_seq_len_train"], :]
                    # Ensure EOS if truncated
                    if not torch.equal(svg_enc_input_pad[-1, :2], eos_token_row_input[:2].to(device)):
                        svg_enc_input_pad[-1] = eos_token_row_input.to(device)
                    svg_attn_mask = torch.zeros(config["max_seq_len_train"], dtype=torch.bool, device=device)
                elif len_svg_unp < config["max_seq_len_train"]:
                    pad_len = config["max_seq_len_train"] - len_svg_unp
                    svg_pad_tensor = padding_row_template_input.unsqueeze(0).repeat(pad_len, 1)
                    svg_enc_input_pad = torch.cat([svg_enc_input_unp, svg_pad_tensor], dim=0)
                    svg_attn_mask = torch.cat([torch.zeros(len_svg_unp, device=device, dtype=torch.bool), torch.ones(pad_len, device=device, dtype=torch.bool)])
                
                svg_batch = svg_enc_input_pad.unsqueeze(0) # Add batch dimension: [1, L_svg, N_features]
                svg_mask_batch = svg_attn_mask.unsqueeze(0) # Add batch dimension: [1, L_svg]

                # --- Prepare ENCODER input for Pixel (Keys/Values) ---
                pixel_batch = final_pixel_patch_tokens_seq.unsqueeze(0) # Add batch dimension: [1, M, D_dino]
                # Pixel mask is all False if M is fixed and no padding is applied to pixel sequences
                pixel_padding_mask_batch = torch.zeros(pixel_batch.shape[1], dtype=torch.bool, device=device).unsqueeze(0) # [1, M]
                
                # --- Forward Pass through Encoder ---
                mu, log_var = model.encoder(svg_batch, pixel_batch, svg_mask_batch, pixel_padding_mask_batch)
                z = mu # For Diffusion Transformer, typically the mean is used as the latent input
                
                # Fetch text caption, default to filename if not found
                text_caption = svg_captions.get(filename_stem, filename_stem)

                z_data_list.append({
                    'filename': Path(file_path).name, # Store only filename for lighter storage
                    'z': z.squeeze(0).cpu().numpy(), # Squeeze batch dim, move to CPU, convert to numpy
                    'text': text_caption
                })
            except Exception as e:
                print(f"Error processing item {filename_stem} (Path: {file_path}): {e}")
                traceback.print_exc()
                
    if z_data_list:
        print(f"\nEncoded {len(z_data_list)} items.")
        zdataset = zDataset(z_data_list)
        print(f"Saving zDataset to '{OUTPUT_Z_DATASET_PATH}'...")
        try:
            torch.save(zdataset, OUTPUT_Z_DATASET_PATH)
            print("zDataset saved successfully.")
        except Exception as e:
            print(f"Error saving zDataset: {e}")
            traceback.print_exc()
    else:
        print("No latent vectors were generated.")

    print("\nScript finished.")
    