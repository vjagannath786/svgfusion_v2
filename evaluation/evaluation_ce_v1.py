# --- START OF FILE evaluate_ce.py ---
import torch
import math
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import traceback
import torch.nn.functional as F

# Ensure sys.path is set if svgutils or dataset_preparation_dynamic is not in standard locations
import sys
import random
sys.path.append(".") # Add current directory to path

try:
    from models.vpvae_accelerate_ce import VPVAE # Your trained model class
    from svgutils import SVGToTensor_Normalized, tensor_to_svg_file_hybrid_wrapper # For de-norm params and vocabs
    from svgutils import load_dino_model_components # To infer DINO dims
    print("Successfully imported VPVAE, SVGToTensor_Normalized, and load_dino_model_components.")
except ImportError as e:
    print(f"CRITICAL ERROR importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Paths and Config ---
# Directory where individual precomputed SVG .pt files are saved
PRECOMPUTED_DATA_OUTPUT_DIR = "precomputed_patch_tokens_data" 
# Path to the file containing a list of paths to the individual .pt files
PRECOMPUTED_FILE_LIST_PATH = "precomputed_patch_tokens_file_list.pt"

# !!! REPLACE WITH YOUR ACTUAL TRAINED HYBRID MODEL PATH !!!
MODEL_PATH = './best_models/vp_vae_accel_hybrid_zany-armadillo-12_s3000_best.pt' 

# Output directory for reconstructed SVGs
OUTPUT_DIR = 'vae_reconstructions_eval_patch_tokens' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Set Seed ---
def set_seed_eval(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed_eval(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Dynamically Infer Configuration Parameters ---
    print("Inferring model configuration from data and DINOv2 model...")

    # Load DINOv2 to get its output dimensions for model config
    # We load DINOv2 temporarily here just to get the dimensions.
    # It will not be moved to GPU unless explicitly done by load_dino_model_components.
    # The returned model/processor are not used for inference in VPVAE directly.
    _, _, _, dino_embed_dim_from_data, fixed_dino_patch_seq_length = load_dino_model_components()
    
    # Load the list of precomputed file paths
    try:
        precomputed_file_paths = torch.load(PRECOMPUTED_FILE_LIST_PATH, map_location='cpu')
        if not precomputed_file_paths: 
            print("Error: Precomputed file list is empty. Cannot infer SVG dimensions."); sys.exit(1)
    except Exception as e:
        print(f"Error loading precomputed file paths from {PRECOMPUTED_FILE_LIST_PATH}: {e}"); traceback.print_exc(); sys.exit(1)

    # Load the first actual precomputed data item to determine SVG tensor dimensions reliably
    try:
        first_precomputed_item_data = torch.load(precomputed_file_paths[0], map_location='cpu')
        num_total_svg_features_from_data = first_precomputed_item_data['full_svg_matrix_content'].shape[1]
        # num_other_continuous_features is the number of continuous columns AFTER elem_id and cmd_id
        num_other_continuous_features_cfg = num_total_svg_features_from_data - 2 
    except Exception as e:
        print(f"Error loading first precomputed item to infer SVG dimensions: {e}"); traceback.print_exc(); sys.exit(1)

    # Instantiate SVGToTensor_Normalized (for vocab sizes and default normalization params)
    # This instance is ONLY to get vocab sizes and default normalization params
    temp_svg_tensor_converter_for_config = SVGToTensor_Normalized() 
    num_element_types_cfg = len(temp_svg_tensor_converter_for_config.ELEMENT_TYPES)
    num_command_types_cfg = len(temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES)
    
    element_pad_idx_cfg = temp_svg_tensor_converter_for_config.ELEMENT_TYPES.get('<PAD>', 0) 
    command_pad_idx_cfg = temp_svg_tensor_converter_for_config.PATH_COMMAND_TYPES.get('NO_CMD', 0)

    # Finalized configuration dictionary
    config = {
        "max_seq_len_train": 1024, # This should match training
        "num_element_types": num_element_types_cfg,
        "num_command_types": num_command_types_cfg,
        "element_embed_dim": 64,  # Use actual from training
        "command_embed_dim": 64,  # Use actual from training
        "num_other_continuous_svg_features": num_other_continuous_features_cfg,
        "pixel_feature_dim": dino_embed_dim_from_data,
        "encoder_d_model": 512,   # Example, use actual from training
        "decoder_d_model": 512,   # Example, use actual from training
        "encoder_layers": 4,      # Example, use actual from training
        "decoder_layers": 4,      # Example, use actual from training
        "num_heads": 8,           # Example, use actual from training
        "latent_dim": 32,         # Example, use actual from training
        "element_padding_idx": element_pad_idx_cfg,
        "command_padding_idx": command_pad_idx_cfg,
        "fixed_dino_patch_seq_length": fixed_dino_patch_seq_length # Inferred
    }
    print(f"Using inferred evaluation config: {config}")

    if MODEL_PATH.startswith('./best_models/vp_vae_accel_hybrid_YOUR_RUN_NAME'): # Placeholder check
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CRITICAL: Update MODEL_PATH to your actual trained hybrid model file. !!!")
        print("!!! The 'config' dictionary above is dynamically inferred, which is good. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # sys.exit(1) # Optionally exit if placeholder path is used

    # --- 2. Load Model State Dictionary ---
    print(f"Loading model state_dict from '{MODEL_PATH}'...")
    try:
        loaded_state_dict = torch.load(MODEL_PATH, map_location=device)
        print("Loaded state_dict successfully.")
    except Exception as e:
        print(f"Error loading state_dict from {MODEL_PATH}: {e}"); traceback.print_exc(); sys.exit(1)
    
    # --- 3. Instantiate SVGToTensor_Normalized (for de-norm info during SVG generation) ---
    # This instance must be consistent with the training's normalization scheme
    svg_tensor_converter_eval = SVGToTensor_Normalized() 
    element_map_rev = {v: k for k, v in svg_tensor_converter_eval.ELEMENT_TYPES.items()}
    command_map_rev = {v: k for k, v in svg_tensor_converter_eval.PATH_COMMAND_TYPES.items()}

    # --- 4. Instantiate VPVAE Model ---
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
    ).to(device)
    
    try: 
        model.load_state_dict(loaded_state_dict)
        print("Model weights applied to instantiated model successfully.")
    except Exception as e: 
        print(f"Error applying loaded state_dict to model: {e}"); traceback.print_exc(); sys.exit(1)
    model.eval()

    # --- 5. Prepare SOS/EOS/PAD Token Rows (for encoder input) ---
    # These must match the exact dimensions used during training
    default_cont_param_values_for_tokens = torch.full(
        (config["num_other_continuous_svg_features"],), 
        svg_tensor_converter_eval.DEFAULT_PARAM_VAL, dtype=torch.float32
    )
    sos_elem_id_val = float(svg_tensor_converter_eval.ELEMENT_TYPES['<BOS>'])
    eos_elem_id_val = float(svg_tensor_converter_eval.ELEMENT_TYPES['<EOS>'])
    pad_elem_id_val = float(config["element_padding_idx"]) 
    no_cmd_id_val = float(config["command_padding_idx"])   
    
    sos_token_row_input = torch.cat([torch.tensor([sos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])
    eos_token_row_input = torch.cat([torch.tensor([eos_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])
    padding_row_template_input = torch.cat([torch.tensor([pad_elem_id_val, no_cmd_id_val], dtype=torch.float32), default_cont_param_values_for_tokens])

    # No longer need sos_pixel_embed_input, pixel_padding_template_input as they are not passed to dataset/dataloader
    # The actual DINOv2 embeddings are loaded from file as part of the item_data.

    # --- 6. Select Samples to Evaluate (now loading paths) ---
    # `precomputed_file_paths` already loaded at the start
    num_samples_to_evaluate = min(5, len(precomputed_file_paths)) 
    eval_file_paths = precomputed_file_paths[:num_samples_to_evaluate] # Now a list of paths
    
    # Example filtering for 'unicorn' - Adjust as needed
    filtered_eval_file_paths = [
        fp for fp in precomputed_file_paths if 'crab' in Path(fp).stem.lower()
    ]
    if filtered_eval_file_paths:
        eval_file_paths = filtered_eval_file_paths
        print(f"Found {len(eval_file_paths)} SVGs with 'unicorn' in filename for evaluation.")
    else:
        print(f"No 'unicorn' SVGs found, using first {len(eval_file_paths)} items from the list.")

    print(f"\nEvaluating VAE Reconstruction for {len(eval_file_paths)} SVGs (Saving to '{OUTPUT_DIR}')...")
    reconstruction_count = 0
    latent_collection = [] # For t-SNE if desired
    filename_labels_for_tsne = [] # For t-SNE if desired

    with torch.no_grad():
        for file_path in tqdm(eval_file_paths, desc="Evaluating SVGs"):
            try:
                # --- MODIFIED: Load item_data on demand ---
                item_data = torch.load(file_path, map_location='cpu') 
                filename_stem = item_data['file_name_stem']
                
                full_svg_content_unpadded = item_data['full_svg_matrix_content'].to(device)
                
                # --- MODIFIED: Get final pixel patch tokens sequence ---
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
                # final_pixel_patch_tokens_seq already has shape [M, D_dino]
                pixel_batch = final_pixel_patch_tokens_seq.unsqueeze(0) # Add batch dimension: [1, M, D_dino]
                # Pixel mask is all False if M is fixed and no padding is applied to pixel sequences
                pixel_padding_mask_batch = torch.zeros(pixel_batch.shape[1], dtype=torch.bool, device=device).unsqueeze(0) # [1, M]

                # --- Forward Pass ---
                mu, log_var = model.encoder(svg_batch, pixel_batch, svg_mask_batch, pixel_padding_mask_batch)
                z = mu # For reconstruction, usually use mu
                
                elem_logits, cmd_logits, cont_params_pred = model.decoder(z, target_len=config["max_seq_len_train"])

                # --- Process Decoder Outputs ---
                pred_elem_ids = elem_logits.squeeze(0).argmax(dim=-1)      
                pred_cmd_ids = cmd_logits.squeeze(0).argmax(dim=-1)        
                
                # Stack continuous parameter logits and get argmax for each param
                param_logits_batch = torch.stack(cont_params_pred, dim=2) # [1, L_svg, P, N_bins]
                pred_bin_indices = param_logits_batch.squeeze(0).argmax(dim=-1) # [L_svg, P] - Keep as LONG             

                reconstructed_tensor_hybrid = torch.cat([
                    pred_elem_ids.long().unsqueeze(-1), pred_cmd_ids.long().unsqueeze(-1), pred_bin_indices.long()
                ], dim=-1).cpu()

                # Determine actual length of reconstructed SVG (up to EOS or PAD)
                actual_recon_len = config["max_seq_len_train"]
                pred_elem_ids_list_cpu = pred_elem_ids.cpu().tolist()
                eos_id_int = int(svg_tensor_converter_eval.ELEMENT_TYPES['<EOS>'])
                pad_id_int = int(config["element_padding_idx"])
                for i_len_rec in range(reconstructed_tensor_hybrid.shape[0]):
                    if pred_elem_ids_list_cpu[i_len_rec] == eos_id_int or pred_elem_ids_list_cpu[i_len_rec] == pad_id_int:
                        actual_recon_len = i_len_rec 
                        break
                
                output_svg_filename = os.path.join(OUTPUT_DIR, f"reconstructed_hybrid_{filename_stem}.svg")
                
                print(f"Reconstructing {filename_stem}. Actual reconstructed length: {actual_recon_len}")
                # Optional: print first few rows of original vs reconstructed for quick comparison
                # print("Original SVG (first 5 rows):")
                # print(full_svg_content_unpadded[:5, :])
                # print("Reconstructed SVG (first 5 rows):")
                # print(reconstructed_tensor_hybrid[:5, :])

                tensor_to_svg_file_hybrid_wrapper(
                    reconstructed_tensor_hybrid, output_svg_filename,
                    svg_tensor_converter_eval,
                    actual_len=actual_recon_len )
                reconstruction_count += 1
                
                # --- For t-SNE visualization if needed (using mu as the latent vector for the whole SVG) ---
                # To represent the entire SVG for t-SNE, you might average mu over its sequence dimension
                # or use the mu corresponding to the <EOS> token if your model learns to aggregate there.
                # Here, we'll average mu for the valid tokens.
                if svg_mask_batch is not None:
                    valid_mask_mu = (~svg_mask_batch[0, :mu.shape[1]]).unsqueeze(-1).float() # [L_svg, 1]
                    pooled_mu = (mu[0] * valid_mask_mu).sum(dim=0) / (valid_mask_mu.sum() + 1e-9) # [latent_dim]
                else:
                    pooled_mu = mu[0].mean(dim=0) # [latent_dim]
                latent_collection.append(pooled_mu.cpu().numpy())
                filename_labels_for_tsne.append(filename_stem)

            except Exception as e_item:
                print(f"ERROR processing item {filename_stem} (Path: {file_path}): {e_item}"); traceback.print_exc()

    print(f"\nFinished evaluation. Reconstructed {reconstruction_count} SVGs in '{OUTPUT_DIR}'.")
    
    # --- t-SNE Plotting (if latent_collection is populated) ---
    if latent_collection and filename_labels_for_tsne and len(latent_collection) > 1:
        print("\nGenerating t-SNE plot for collected latent representations...")
        try:
            latent_vectors_np = np.vstack(latent_collection)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(latent_vectors_np)-1)) # Adjust perplexity
            tsne_results = tsne.fit_transform(latent_vectors_np)

            plt.figure(figsize=(12, 10))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
            for i, txt in enumerate(filename_labels_for_tsne):
                plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]), textcoords="offset points", xytext=(0,5), ha='center')
            plt.title('t-SNE of Latent Space')
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, "latent_tsne_plot.png"))
            print(f"Saved t-SNE plot to '{os.path.join(OUTPUT_DIR, 'latent_tsne_plot.png')}'")
        except Exception as e_tsne:
            print(f"Error generating t-SNE plot: {e_tsne}"); traceback.print_exc()
    elif latent_collection:
        print("Skipping t-SNE: Not enough unique latent vectors (need > 1) or perplexity issue.")


# --- END OF FILE evaluate_ce.py ---
