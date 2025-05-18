# --- START OF FILE evaluate_vp_vae_hybrid_final.py ---
import torch
import math
import os
from pathlib import Path
from tqdm import tqdm # Optional, for processing multiple files
import numpy as np # For t-SNE if used
from sklearn.manifold import TSNE # For t-SNE if used
import matplotlib.pyplot as plt # For t-SNE if used
import traceback # For detailed errors


# Ensure sys.path is set if svgutils is not in standard locations
import sys
import random
sys.path.append(".")

try:
    from models import VPVAE # Your trained model class
    from svgutils import SVGToTensor_Normalized, tensor_to_svg_file_hybrid_wrapper # For de-norm params and vocabs
    print("Successfully imported VPVAE and SVGToTensor_Normalized.")
except ImportError as e:
    print(f"CRITICAL ERROR importing VPVAE or SVGToTensor_Normalized: {e}")
    sys.exit(1)


# --- Paths and Config ---
# Path to the precomputed data list (output of dataset_preparation_dynamic.py using HYBRID format)
PRECOMPUTED_DATA_PATH = './datasets/optimized_progressive_dataset_precomputed_v2.pt' # ADJUST IF NEEDED
MODEL_PATH = 'vp_vae_accel_hybrid_ancient-dust-2_s5000_best.pt' # !!! REPLACE WITH YOUR ACTUAL HYBRID MODEL PATH !!!
OUTPUT_DIR = 'vae_reconstructions_hybrid_eval_final' # New output dir name
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Set Seed ---
def set_seed_eval(seed): # Renamed to avoid conflict if imported elsewhere
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed_eval(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Define Config Manually (MUST MATCH THE TRAINED HYBRID MODEL'S CONFIG) ---
    print("Defining model configuration for evaluation...")
    # This instance is ONLY to get vocab sizes and default normalization params
    # It assumes its internal settings (like ELEMENT_TYPES, PATH_COMMAND_TYPES, COORD_MIN/MAX etc.)
    # are consistent with what was used for training the loaded model.
    temp_svg_tensor_converter_for_config = SVGToTensor_Normalized() 
    
    num_element_types_cfg = len(temp_svg_tensor_converter_for_config.ELEMENT_TYPES)
    num_command_types_cfg = len(temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES)
    # Number of continuous features *excluding* element ID and command ID
    # This should be 12 if it's 8 geo params + 4 style params
    num_other_continuous_features_cfg = temp_svg_tensor_converter_for_config.num_geom_params + temp_svg_tensor_converter_for_config.num_fill_style_params
    
    element_pad_idx_cfg = temp_svg_tensor_converter_for_config.ELEMENT_TYPES.get('<PAD>', 0) 
    command_pad_idx_cfg = temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES.get('NO_CMD', 0)

    # !!! REPLACE THESE WITH THE ACTUAL VALUES FROM YOUR TRAINED MODEL'S wandb.config !!!
    config = {
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
    }
    print(f"Using evaluation config: {config}")
    if MODEL_PATH == 'vp_vae_accel_hybrid_your_run_name_sXXXX_best.pt': # Default check
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL: Update MODEL_PATH to your actual trained hybrid model file. !!!")
        print("!!! Also, ensure the 'config' dictionary above matches that model.       !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # sys.exit(1) # Optionally exit if placeholder path is used

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
        precomputed_data_list_for_eval = torch.load(PRECOMPUTED_DATA_PATH, map_location='cpu')
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

    # --- Files/Samples to Evaluate ---
    num_samples_to_evaluate = min(5, len(precomputed_data_list_for_eval)) 
    eval_items_list = precomputed_data_list_for_eval[:num_samples_to_evaluate]
    
    print(f"\nEvaluating VAE Reconstruction for {len(eval_items_list)} SVGs (Saving to '{OUTPUT_DIR}')...")
    reconstruction_count = 0; latent_collection = []; filename_labels_for_tsne = []

    ## if umbrella in precomputed_data_list_for_eval[0]['file_name_stem']:

    # Filter for items with 'umbrella' in the filename stem
    umbrella_items = [item for item in precomputed_data_list_for_eval if 'crab' in item['file_name_stem']]
    if umbrella_items:
        eval_items_list = umbrella_items
        print(f"Found {len(eval_items_list)} SVGs with 'umbrella' in filename.")
    else:
        num_samples_to_evaluate = min(5, len(precomputed_data_list_for_eval))
        eval_items_list = precomputed_data_list_for_eval[:num_samples_to_evaluate]
        print(f"No 'umbrella' SVGs found, using first {len(eval_items_list)} items.")




    with torch.no_grad():
        for item_data in tqdm(eval_items_list, desc="Evaluating SVGs"):
            try:
                filename_stem = item_data['file_name_stem']
                
                full_svg_content_unpadded = item_data['full_svg_matrix_content'].to(device) 
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
                elem_logits, cmd_logits, cont_params_pred = model.decoder(z, target_len=config["max_seq_len_train"])

                pred_elem_ids = elem_logits.squeeze(0).argmax(dim=-1)      
                pred_cmd_ids = cmd_logits.squeeze(0).argmax(dim=-1)        
                pred_cont_params = cont_params_pred.squeeze(0)             

                reconstructed_tensor_hybrid = torch.cat([
                    pred_elem_ids.float().unsqueeze(-1), pred_cmd_ids.float().unsqueeze(-1), pred_cont_params
                ], dim=-1).cpu()

                actual_recon_len = config["max_seq_len_train"]
                pred_elem_ids_list_cpu = pred_elem_ids.cpu().tolist()
                eos_id_int = int(svg_tensor_converter_eval.ELEMENT_TYPES['<EOS>'])
                pad_id_int = int(config["element_padding_idx"])
                for i_len_rec in range(reconstructed_tensor_hybrid.shape[0]):
                    if pred_elem_ids_list_cpu[i_len_rec] == eos_id_int or pred_elem_ids_list_cpu[i_len_rec] == pad_id_int:
                        actual_recon_len = i_len_rec 
                        break
                
                output_svg_filename = os.path.join(OUTPUT_DIR, f"reconstructed_hybrid_{filename_stem}.svg")
                print(actual_recon_len)
                print(reconstructed_tensor_hybrid.shape)
                print(reconstructed_tensor_hybrid[:7])

                print(actual_recon_len)
                
                
                tensor_to_svg_file_hybrid_wrapper(
                    reconstructed_tensor_hybrid, output_svg_filename,
                    svg_tensor_converter_eval,
                    actual_len=actual_recon_len )
                reconstruction_count += 1
                # latent_collection.append(mu.squeeze(0).cpu().numpy())
                # filename_labels_for_tsne.append(filename_stem)
            except Exception as e_item:
                print(f"ERROR processing item {filename_stem}:"); traceback.print_exc()

    print(f"\nFinished evaluation. Reconstructed {reconstruction_count} SVGs in '{OUTPUT_DIR}'.")
    # ... (t-SNE plotting as before) ...
    if latent_collection and filename_labels_for_tsne:
        # ... (t-SNE code as before) ...
        pass # Placeholder
