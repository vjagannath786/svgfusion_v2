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
sys.path.append("/home/")

try:
    from models import VPVAE # Your trained model class
    from svgutils import SVGToTensor_Normalized # For de-norm params and vocabs
    print("Successfully imported VPVAE and SVGToTensor_Normalized.")
except ImportError as e:
    print(f"CRITICAL ERROR importing VPVAE or SVGToTensor_Normalized: {e}")
    sys.exit(1)


class TensorToSVGHybrid:
    def __init__(self, svg_tensor_converter_instance):
        self.converter = svg_tensor_converter_instance # Instance of SVGToTensor_Normalized

        # Reverse mappings for convenience
        self.element_map_rev = {v: k for k, v in self.converter.ELEMENT_TYPES.items()}
        self.command_map_rev = {v: k for k, v in self.converter.PATH_COMMAND_TYPES.items()}
        
        # Command IDs from the converter for easier reference
        self.CMD_M = self.converter.PATH_COMMAND_TYPES.get('m', -1) # Use .get for safety
        self.CMD_L = self.converter.PATH_COMMAND_TYPES.get('l', -1)
        self.CMD_H = self.converter.PATH_COMMAND_TYPES.get('h', -1)
        self.CMD_V = self.converter.PATH_COMMAND_TYPES.get('v', -1)
        self.CMD_C = self.converter.PATH_COMMAND_TYPES.get('c', -1)
        self.CMD_S = self.converter.PATH_COMMAND_TYPES.get('s', -1) # Not explicitly handled below, add if needed
        self.CMD_Q = self.converter.PATH_COMMAND_TYPES.get('q', -1)
        self.CMD_T = self.converter.PATH_COMMAND_TYPES.get('t', -1) # Not explicitly handled below, add if needed
        self.CMD_A = self.converter.PATH_COMMAND_TYPES.get('a', -1)
        self.CMD_Z = self.converter.PATH_COMMAND_TYPES.get('z', -1)
        self.CMD_DEF = self.converter.PATH_COMMAND_TYPES.get('DEF', -1) # For shape definitions
        # Assuming SOS/EOS/PAD are handled by their element types, not command types here for path data

    def _denormalize(self, norm_value, val_min, val_max):
        """De-normalizes value from [target_norm_min, target_norm_max] to [val_min, val_max]."""
        target_min = self.converter.target_norm_min
        target_max = self.converter.target_norm_max

        if target_max == target_min: return target_min # Avoid division by zero
        if val_max == val_min: return val_min # If original range was a point

        # De-normalize from [target_min, target_max] to [0, 1]
        norm_0_1 = (norm_value - target_min) / (target_max - target_min)
        # Scale from [0, 1] to [val_min, val_max]
        value = norm_0_1 * (val_max - val_min) + val_min
        return value

    def _format_geo_params(self, cmd_id, geo_params_norm):
        # geo_params_norm is a tensor/list of 8 normalized values (µ0 to ν3)
        # Returns a list of de-normalized, formatted strings for the SVG path 'd' attribute
        params_str = []
        cmd_char = self.command_map_rev.get(cmd_id, '?').lower() # Get char like 'm', 'l'

        # Dequantize based on known parameter meanings for each command
        # µ0,ν0 (params 0,1), µ1,ν1 (2,3), µ2,ν2 (4,5), µ3,ν3 (6,7)
        if cmd_char in ['m', 'l', 't']: # End point (x,y)
            params_str.append(f"{self._denormalize(geo_params_norm[6], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            params_str.append(f"{self._denormalize(geo_params_norm[7], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
        elif cmd_char in ['h']: # End x
            params_str.append(f"{self._denormalize(geo_params_norm[6], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
        elif cmd_char in ['v']: # End y
            params_str.append(f"{self._denormalize(geo_params_norm[7], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
        elif cmd_char in ['c']: # cx1,cy1, cx2,cy2, ex,ey
            for i in range(2, 8): # indices 2 through 7 of geo_params_norm
                params_str.append(f"{self._denormalize(geo_params_norm[i], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
        elif cmd_char in ['q', 's']: # cx,cy, ex,ey (S might need different handling if it implies prev control point)
            # For Q: µ1,ν1 (c1x,c1y), µ3,ν3 (ex,ey) -> params 2,3,6,7
            params_str.append(f"{self._denormalize(geo_params_norm[2], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            params_str.append(f"{self._denormalize(geo_params_norm[3], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            params_str.append(f"{self._denormalize(geo_params_norm[6], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            params_str.append(f"{self._denormalize(geo_params_norm[7], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
        elif cmd_char == 'a': # rx,ry, x-axis-rot, large-arc, sweep, ex,ey
            # Mapping from paper/SVG spec to 8 slots:
            # µ1(rx), ν1(ry), µ2(xrot), ν2(large-arc), µ0(sweep-flag - not used in 8 params?), ν0(?), µ3(ex), ν3(ey)
            # This needs careful mapping based on how SVGToTensor_Normalized packs Arc params into 8 slots.
            # Assuming: geo_params_norm[2]=rx, [3]=ry, [4]=xrot, [5]=large-arc, [0]=sweep (example)
            params_str.append(f"{self._denormalize(geo_params_norm[2], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX):.1f}") # rx
            params_str.append(f"{self._denormalize(geo_params_norm[3], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX):.1f}") # ry
            params_str.append(f"{self._denormalize(geo_params_norm[4], self.converter.ROT_MIN, self.converter.ROT_MAX):.1f}")      # x-axis-rotation
            params_str.append(f"{int(round(self._denormalize(geo_params_norm[5], self.converter.FLAG_MIN, self.converter.FLAG_MAX)))}") # large-arc-flag
            params_str.append(f"{int(round(self._denormalize(geo_params_norm[0], self.converter.FLAG_MIN, self.converter.FLAG_MAX)))}") # sweep-flag (example slot)
            params_str.append(f"{self._denormalize(geo_params_norm[6], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}") # ex
            params_str.append(f"{self._denormalize(geo_params_norm[7], self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}") # ey
        elif cmd_char == 'z':
            pass # No params
        return params_str

    def _format_style_attributes(self, style_params_norm):
        # style_params_norm: [R, G, B, Alpha] (normalized)
        attrs = []
        try:
            r = int(round(self._denormalize(style_params_norm[0], self.converter.COLOR_MIN, self.converter.COLOR_MAX)))
            g = int(round(self._denormalize(style_params_norm[1], self.converter.COLOR_MIN, self.converter.COLOR_MAX)))
            b = int(round(self._denormalize(style_params_norm[2], self.converter.COLOR_MIN, self.converter.COLOR_MAX)))
            alpha = self._denormalize(style_params_norm[3], self.converter.OPACITY_MIN, self.converter.OPACITY_MAX)

            r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
            alpha = max(0.0, min(1.0, alpha))

            attrs.append(f'fill="#{r:02x}{g:02x}{b:02x}"')
            if alpha < 0.995: # Only add opacity if not fully opaque
                attrs.append(f'fill-opacity="{alpha:.2f}"')
            # Assuming no stroke for simplicity here, add if your model predicts stroke
            attrs.append('stroke="none"')
        except IndexError:
            print("WARN: Index error processing style params.")
            attrs.append('fill="black" stroke="none"') # Fallback
        return " ".join(attrs)

    def reconstruct_svg_elements(self, tensor_data_hybrid, actual_len=None):
        # tensor_data_hybrid: [L, 2 (IDs) + N_other_cont_params]
        # Example: L, 2 + 8 geo + 4 style = L, 14
        if actual_len is not None:
            tensor_to_process = tensor_data_hybrid[:actual_len]
        else:
            tensor_to_process = tensor_data_hybrid
        
        svg_elements_strings = []
        current_path_segments = []
        last_element_type_id = -1

        for i in range(tensor_to_process.shape[0]):
            row = tensor_to_process[i]
            elem_id = int(row[0].item())
            cmd_id = int(row[1].item())
            # Continuous params start from index 2
            # First 8 are geo, next 4 are style (R,G,B,A)
            geo_params_norm = row[2 : 2+8] 
            style_params_norm = row[2+8 : 2+8+4]

            elem_tag_str = self.element_map_rev.get(elem_id, None)
            cmd_char_from_id = self.command_map_rev.get(cmd_id, '').lower() # m, l, def, etc.

            if elem_tag_str is None or elem_tag_str in ['<BOS>', '<PAD>']:
                continue
            if elem_tag_str == '<EOS>':
                break # Stop processing at EOS

            # Finalize previous path if current element is not a path continuation
            if elem_tag_str != "<path>" and current_path_segments:
                style_attrs_str = self._format_style_attributes(last_path_style_params) # Use style of last path segment
                svg_elements_strings.append(f'  <path d="{" ".join(current_path_segments)}" {style_attrs_str}/>')
                current_path_segments = []

            if elem_tag_str == "<path>":
                if not current_path_segments: # Start of a new path element
                    # We need to decide when a path element "ends". 
                    # The original paper implies paths can have multiple commands then a style.
                    # For simplicity, let's assume each path command row might start a new path
                    # if the previous wasn't also a path, or continues it.
                    # This part of the logic is tricky without knowing how SVGToTensor groups path commands
                    # into a single <path> element with one style.
                    # Let's assume for now that a sequence of path commands belong to one path until a non-path element.
                    pass # Handled by appending to current_path_segments

                formatted_geo_params = self._format_geo_params(cmd_id, geo_params_norm)
                current_path_segments.append(f"{cmd_char_from_id.upper()} {' '.join(formatted_geo_params)}")
                last_path_style_params = style_params_norm # Store style for when path ends
            
            elif elem_tag_str == "<rect>" and cmd_char_from_id == "def":
                # Assuming geo_params_norm for rect: x,y,rx,ry,w,h (in the first 6 slots of 8)
                x = self._denormalize(geo_params_norm[0], self.converter.COORD_MIN, self.converter.COORD_MAX)
                y = self._denormalize(geo_params_norm[1], self.converter.COORD_MIN, self.converter.COORD_MAX)
                rx = self._denormalize(geo_params_norm[2], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                ry = self._denormalize(geo_params_norm[3], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                w = self._denormalize(geo_params_norm[4], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                h = self._denormalize(geo_params_norm[5], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                style_attrs_str = self._format_style_attributes(style_params_norm)
                svg_elements_strings.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" {style_attrs_str}/>')

            elif elem_tag_str == "<circle>" and cmd_char_from_id == "def":
                # Assuming geo_params_norm for circle: cx,cy,r (in first 3 slots)
                cx = self._denormalize(geo_params_norm[0], self.converter.COORD_MIN, self.converter.COORD_MAX)
                cy = self._denormalize(geo_params_norm[1], self.converter.COORD_MIN, self.converter.COORD_MAX)
                r  = self._denormalize(geo_params_norm[2], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                style_attrs_str = self._format_style_attributes(style_params_norm)
                svg_elements_strings.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {style_attrs_str}/>')
            
            elif elem_tag_str == "<ellipse>" and cmd_char_from_id == "def":
                # Assuming geo_params_norm for ellipse: cx,cy,rx,ry (in first 4 slots)
                cx = self._denormalize(geo_params_norm[0], self.converter.COORD_MIN, self.converter.COORD_MAX)
                cy = self._denormalize(geo_params_norm[1], self.converter.COORD_MIN, self.converter.COORD_MAX)
                rx = self._denormalize(geo_params_norm[2], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                ry = self._denormalize(geo_params_norm[3], self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                style_attrs_str = self._format_style_attributes(style_params_norm)
                svg_elements_strings.append(f'  <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" {style_attrs_str}/>')

            last_element_type_id = elem_id

        # Finalize any pending path after loop
        if current_path_segments:
            style_attrs_str = self._format_style_attributes(last_path_style_params)
            svg_elements_strings.append(f'  <path d="{" ".join(current_path_segments)}" {style_attrs_str}/>')

        return svg_elements_strings

    def create_svg_document(self, svg_element_strings_list, viewbox_dims=(128,128)):
        width, height = viewbox_dims
        viewBox_str = f"0 0 {width} {height}"
        svg_header = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox_str}">'
        svg_footer = '</svg>'
        return f"{svg_header}\n" + "\n".join(svg_element_strings_list) + f"\n{svg_footer}"

# --- Wrapper Function ---
def tensor_to_svg_file_hybrid_wrapper(
    predicted_hybrid_tensor_cpu, # Tensor from model output, moved to CPU
    output_filename,
    svg_tensor_converter_instance, # Passed for de-norm data
    actual_len=None # Optional: number of valid rows before padding/EOS
):
    reconstructor = TensorToSVGHybrid(svg_tensor_converter_instance)
    svg_element_strings = reconstructor.reconstruct_svg_elements(predicted_hybrid_tensor_cpu, actual_len)
    # Assuming viewbox from training config or a default for reconstruction
    # You might need to pass viewbox info if it varies and is important
    viewbox_w = getattr(svg_tensor_converter_instance, 'viewBox_width_norm', 128) # Example default
    viewbox_h = getattr(svg_tensor_converter_instance, 'viewBox_height_norm', 128)
    
    final_svg_content = reconstructor.create_svg_document(svg_element_strings, viewbox_dims=(viewbox_w, viewbox_h))
    
    try:
        with open(output_filename, 'w') as f:
            f.write(final_svg_content)
        print(f"INFO: Reconstructed SVG saved to: {output_filename}")
    except Exception as e_write:
        print(f"ERROR: Failed to write SVG file {output_filename}: {e_write}")
    return final_svg_content

# --- Paths and Config ---
# Path to the precomputed data list (output of dataset_preparation_dynamic.py using HYBRID format)
PRECOMPUTED_DATA_PATH = '/home/dataset/optimized_progressive_dataset_precomputed_v2.pt' # ADJUST IF NEEDED
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
    umbrella_items = [item for item in precomputed_data_list_for_eval if 'email-icon' in item['file_name_stem']]
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
                print(reconstructed_tensor_hybrid.shape)
                print(reconstructed_tensor_hybrid[:actual_recon_len])

                #print(actual_recon_len)
                
                
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
