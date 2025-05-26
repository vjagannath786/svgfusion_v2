import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import sys
import traceback

sys.path.append("/home/svgfusion_v2/")

try:
    from models import VPVAE, set_seed
    from svgutils import SVGToTensor_Normalized
except ImportError as e:
    print(f"CRITICAL ERROR importing VPVAE or SVGToTensor_Normalized: {e}")
    sys.exit(1)
    
PRECOMPUTED_DATA_PATH = '/home/svgfusion_v2/dataset/optimized_progressive_dataset_precomputed_v8.pt' 
MODEL_PATH = '/home/model_weights/vp_vae_accel_hybrid_light-feather-36_s5000_best.pt'
OUTPUT_Z_DATASET_PATH = 'zdataset_vpvae.pt'


import json

with open('/home/svg_captions.json', 'r') as f:
    svg_captions = json.load(f)
    

# --- zDataset Definition ---
class zDataset(Dataset):
    def __init__(self, z_data_list):
        # z_data_list is a list of dicts: [{'filename': str, 'z': tensor}, ...]
        self.z_data = z_data_list
        # Calculate statistics once
        all_z = torch.stack([torch.tensor(item['z']) for item in z_data_list])
        self.z_mean = all_z.mean(0, keepdim=True)
        #print(self.z_mean)
        self.z_std = all_z.std(0, keepdim=True)
    def __len__(self):
        return len(self.z_data)
    def __getitem__(self, idx):
        item = self.z_data[idx]
        filename = item['filename'].split(".svg")[0]
        text = svg_captions.get(filename, filename)
        print(text)
        z = item['z']
        return {'z': z, 'text' : text, 'filename': item['filename']}


# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    print(f"Using device: {device}")
    
    print("Defining model configuration for evaluation...")
    
    temp_svg_tensor_converter_for_config = SVGToTensor_Normalized() 
    
    num_element_types_cfg = len(temp_svg_tensor_converter_for_config.ELEMENT_TYPES)
    num_command_types_cfg = len(temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES)
    
    num_other_continuous_features_cfg = temp_svg_tensor_converter_for_config.num_geom_params + temp_svg_tensor_converter_for_config.num_fill_style_params
    
    element_pad_idx_cfg = temp_svg_tensor_converter_for_config.ELEMENT_TYPES.get('<PAD>', 0) 
    command_pad_idx_cfg = temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES.get('NO_CMD', 0)

    ## PRECHECK this with triaining config
    config = {
        "max_seq_len_train": 1024,
        "num_element_types": 7,
        "num_command_types": 14,
        "element_embed_dim": 64,  
        "command_embed_dim": 64,  
        "num_other_continuous_svg_features": 12,
        "pixel_feature_dim": 384, 
        "encoder_d_model": 512,   
        "decoder_d_model": 512,   
        "encoder_layers": 4,      
        "decoder_layers": 4,      
        "num_heads": 8,           
        "latent_dim": 128,        
        "element_padding_idx": 6,
        "command_padding_idx": 0,
    }
    print(f"Using evaluation config: {config}")
    
    # --- 2. Load Model State Dictionary ---
    print(f"Loading model state_dict from '{MODEL_PATH}'...")
    try:
        loaded_state_dict = torch.load(MODEL_PATH, map_location=device)
        print("Loaded state_dict successfully.")
    except Exception as e:
        print(f"Error loading state_dict from {MODEL_PATH}: {e}"); traceback.print_exc(); exit()
        
    
    # --- 3. Instantiate SVGToTensor_Normalized (for de-norm info during SVG generation) ---
    svg_tensor_converter_eval = SVGToTensor_Normalized() # For de-normalization
    element_map_rev = {v: k for k, v in svg_tensor_converter_eval.ELEMENT_TYPES.items()}
    command_map_rev = {v: k for k, v in svg_tensor_converter_eval.PATH_COMMAND_TYPES.items()}
    
    # --- 4. Instantiate HYBRID Model ---
    print("Instantiating Hybrid VPVAE model...")
    model = VPVAE( # This is your hybrid VPVAE
        num_element_types=config["num_element_types"],
        num_command_types=config["num_command_types"],
        element_embed_dim=config["element_embed_dim"],
        command_embed_dim=config["command_embed_dim"],
        num_other_continuous_svg_features=config["num_other_continuous_svg_features"],
        num_other_continuous_params_to_reconstruct=config["num_other_continuous_svg_features"], # Decoder outputs same N continuous
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
    ).to(device)
    try: 
        model.load_state_dict(loaded_state_dict)
        print("Model weights applied to instantiated model successfully.")
    except Exception as e: 
        print(f"Error applying loaded state_dict to model: {e}"); traceback.print_exc(); exit()
    model.eval()
    
    # --- 5. Load Precomputed Data List (for inputs) ---
    print(f"Loading precomputed data list for inputs from '{PRECOMPUTED_DATA_PATH}'...")
    try:
        precomputed_data_list_for_eval = torch.load(PRECOMPUTED_DATA_PATH, map_location=device)
        if not precomputed_data_list_for_eval: print("Warning: Precomputed data list for eval is empty.")
    except Exception as e: print(f"Error loading precomputed data: {e}"); traceback.print_exc(); exit()
    
    # --- SOS/EOS Token Rows (for encoder input) ---
    # First two cols are float(ID), rest are normalized continuous
    sos_elem_id_val = float(svg_tensor_converter_eval.ELEMENT_TYPES['<BOS>'])
    eos_elem_id_val = float(svg_tensor_converter_eval.ELEMENT_TYPES['<EOS>'])
    # Use the actual padding ID from config for consistency
    pad_elem_id_val = float(config["element_padding_idx"]) 
    no_cmd_id_val = float(config["command_padding_idx"])   
    
    default_cont_param_values_for_tokens = torch.full(
        (config["num_other_continuous_svg_features"],), 
        svg_tensor_converter_eval.DEFAULT_PARAM_VAL, dtype=torch.float32
    )
    sos_token_row_input = torch.cat([torch.tensor([sos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])
    eos_token_row_input = torch.cat([torch.tensor([eos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])
    
    sos_pixel_embed_input = torch.zeros((1, config["pixel_feature_dim"]), dtype=torch.float32) # For encoder input
    z_data_list = []
    with torch.no_grad():
        for item_data in tqdm(precomputed_data_list_for_eval, desc="Preparing Latents"):
            try:
                filename_stem = item_data['file_name_stem']
                
                full_svg_content_unpadded = item_data['full_svg_matrix_content'].to(device)
                #print(full_svg_content_unpadded) 
                final_pixel_cls_token = item_data['cumulative_pixel_CLS_tokens_aligned'][-1, :].to(device)

                # Prepare ENCODER input
                svg_enc_input_unp = torch.cat([
                    sos_token_row_input.unsqueeze(0).to(device), full_svg_content_unpadded, eos_token_row_input.unsqueeze(0).to(device)
                ], dim=0)
                len_svg_unp = svg_enc_input_unp.shape[0]
                
                svg_enc_input_pad = svg_enc_input_unp.clone()
                attn_mask = torch.zeros(len_svg_unp, dtype=torch.bool, device=device)

                if len_svg_unp > config["max_seq_len_train"]:
                    svg_enc_input_pad = svg_enc_input_unp[:config["max_seq_len_train"], :]
                    if not torch.equal(svg_enc_input_pad[-1, :2], eos_token_row_input[:2]):
                        svg_enc_input_pad[-1] = eos_token_row_input.to(device)
                    attn_mask = torch.zeros(config["max_seq_len_train"], dtype=torch.bool, device=device)
                elif len_svg_unp < config["max_seq_len_train"]:
                    pad_len = config["max_seq_len_train"] - len_svg_unp
                    padding_val_svg_row = torch.cat([
                        torch.tensor([pad_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens
                    ]).to(device)
                    svg_pad_tensor = padding_val_svg_row.unsqueeze(0).repeat(pad_len, 1)
                    svg_enc_input_pad = torch.cat([svg_enc_input_unp, svg_pad_tensor], dim=0)
                    attn_mask = torch.cat([torch.zeros(len_svg_unp, device=device, dtype=torch.bool), torch.ones(pad_len, device=device, dtype=torch.bool)])
                
                svg_batch = svg_enc_input_pad.unsqueeze(0)
                svg_mask_batch = attn_mask.unsqueeze(0)

                pix_content_rows = final_pixel_cls_token.unsqueeze(0).repeat(full_svg_content_unpadded.shape[0], 1)
                pix_enc_input_unp = torch.cat([sos_pixel_embed_input.to(device), pix_content_rows, final_pixel_cls_token.unsqueeze(0)], dim=0)
                pix_enc_input_pad = pix_enc_input_unp.clone()
                if len_svg_unp > config["max_seq_len_train"]: # Use len_svg_unp for consistency
                    pix_enc_input_pad = pix_enc_input_unp[:config["max_seq_len_train"], :]
                    if not torch.equal(pix_enc_input_pad[-1,:], final_pixel_cls_token): pix_enc_input_pad[-1,:] = final_pixel_cls_token
                elif len_svg_unp < config["max_seq_len_train"]:
                    pad_len = config["max_seq_len_train"] - len_svg_unp
                    pix_pad_tensor = torch.zeros(pad_len, config["pixel_feature_dim"], device=device)
                    pix_enc_input_pad = torch.cat([pix_enc_input_unp, pix_pad_tensor], dim=0)
                pix_batch = pix_enc_input_pad.unsqueeze(0)
                
                
                mu, log_var = model.encoder(svg_batch, pix_batch, svg_mask_batch, svg_mask_batch)
                z = mu
                #print(f"z shape: {z.squeeze(0).shape}")
                
                z_data_list.append({
                    'filename': filename_stem,
                    'z': z.squeeze(0).cpu().detach()
                })
            except Exception as e:
                print(f"Error processing item {filename_stem}: {e}")
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
    else:
        print("No latent vectors were generated.")

    print("\nScript finished.")
                
                
    
    
    
    