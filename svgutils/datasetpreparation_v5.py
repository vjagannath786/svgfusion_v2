import sys
sys.path.append("/Users/varun_jagannath/Documents/D/svgfusion_v2")
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import traceback
from io import BytesIO
import cairosvg
import numpy as np
import os
from pathlib import Path

# Assuming these are available in your path or defined locally
from svgutils import SVGToTensor_Normalized # MUST be the version outputting fully normalized floats
from svgutils import SVGParser

# --- DINOv2 Loading Utility ---
def load_dino_model_components():
    model_name = 'facebook/dinov2-small'
    print(f"Loading DINOv2 components: {model_name}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        model.eval() # Set model to evaluation mode
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
        print(f"DINOv2 components loaded successfully. Using device: {device}")

        # Determine DINOv2 output sequence length and embedding dimension
        dummy_image = Image.new('RGB', (224, 224)) 
        with torch.no_grad():
            inputs = processor(images=dummy_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            dino_seq_length = outputs.last_hidden_state.shape[1] # M (e.g., 257)
            dino_embed_dim = outputs.last_hidden_state.shape[2] # D_dino (e.g., 384 for dinov2-small)
            
        print(f"DINOv2 output sequence length (M): {dino_seq_length}, embedding dimension (D_dino): {dino_embed_dim}")
        return model, processor, device, dino_embed_dim, dino_seq_length # Return M as well
    except Exception as e:
        print(f"Error loading DINOv2 components: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- SVG String Parsing Helpers (from SVGPathRasterizerAligned) ---
def _create_cumulative_svg_string_static(element_strings, viewinfo):
    elements_combined = "\n".join(element_strings)
    vb_w = viewinfo.get('viewBox',{}).get('width', viewinfo.get('width',100))
    vb_h = viewinfo.get('viewBox',{}).get('height', viewinfo.get('height',100))
    vb_x = viewinfo.get('viewBox',{}).get('x', 0)
    vb_y = viewinfo.get('viewBox',{}).get('y', 0)
    img_w = max(1, int(float(viewinfo.get('width', vb_w))))
    img_h = max(1, int(float(viewinfo.get('height', vb_h))))
    svg_content = f'''<svg width="{img_w}" height="{img_h}"
                     xmlns="http://www.w3.org/2000/svg" viewBox="{vb_x} {vb_y} {vb_w} {vb_h}">
                     {elements_combined}
                     </svg>'''
    return svg_content

def _rasterize_and_embed_static(svg_content_string, viewinfo, dino_model, dino_processor, device):
    try:
        # Match DINOv2 input size for consistent patch token sequence length
        output_w = max(1, int(float(viewinfo.get('width', 224)))) 
        output_h = max(1, int(float(viewinfo.get('height', 224))))
        png_buffer = BytesIO()
        cairosvg.svg2png(bytestring=svg_content_string.encode('utf-8'), output_width=output_w, output_height=output_h, write_to=png_buffer)
        png_buffer.seek(0)
        image = Image.open(png_buffer).convert("RGB")
    except Exception as e: 
        print(f"  Error rasterizing SVG: {e}")
        return None
    try:
        with torch.no_grad():
            inputs = dino_processor(images=image, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0) # [1, M, D_dino] -> [M, D_dino]
            return embedding.cpu()
    except Exception as e: 
        print(f"  Error getting DINO embedding: {e}")
        return None

def parse_element_to_string(element_data): # Simplified combined parser
    el_type = element_data['type']
    style_str_parts = []
    style = element_data.get('style', {})
    for k, v in style.items():
        if k == 'opacity' and v is not None:
            v = str(float(1)) 
        if v is not None and str(v).lower() != 'none':
            style_str_parts.append(f'{k}="{v}"')
    style_string = " " + " ".join(style_str_parts) if style_str_parts else ""

    if el_type == 'path':
        path_str = '<path d="'
        d_parts = []
        for cmd_detail in element_data.get("commands", []):
            parts = cmd_detail["original"].replace(",", " ").split()
            if len(parts) > 5 and ("a" in parts[0].lower() or "A" in parts[0].lower()): 
                try: 
                    parts[2] = str(int(float(parts[2]))); parts[3] = str(int(float(parts[3]))); parts[4] = str(int(float(parts[4])))
                except ValueError: print(f"  Warning: Could not parse arc flags/rot in {parts}")
            d_parts.append(" ".join(parts))
        path_str += " ".join(d_parts) + '"' + style_string + '/>'
        return path_str
    
    elif el_type == 'circle':
        attrs = {item['command']: item['values'] for item in element_data.get('commands', {})}
        cx = attrs.get('cx', 0); cy = attrs.get('cy', 0); r = attrs.get('r', 0)
        return f'<circle cx="{cx}" cy="{cy}" r="{r}"{style_string}/>'
    
    elif el_type == 'rect':
        attrs = {item['command']: item['values'] for item in element_data.get('commands', {})}
        attr_str_parts = [f'{k}="{v}"' for k, v in attrs.items()]
        return f'<rect {" ".join(attr_str_parts)}{style_string}/>'

    elif el_type == 'ellipse':
        attrs = {item['command']: item['values'] for item in element_data.get('commands', {})}
        cx = attrs.get('cx',0); cy = attrs.get('cy',0); rx = attrs.get('rx',0); ry = attrs.get('ry',0)
        return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"{style_string}/>'
    else:
        return ""


# --- Preprocessing Function for a Single SVG ---
def preprocess_svg_for_dynamic_storage(
    svg_file_path, svg_tensor_converter, svg_parser,
    dino_model, dino_processor, dino_device, dino_embed_dim, fixed_dino_patch_seq_length,
    output_dir # New argument: directory to save preprocessed data
):
    file_name_stem = Path(svg_file_path).stem
    output_path = Path(output_dir) / f"{file_name_stem}.pt" # Path to save this SVG's preprocessed data

    # Skip if already processed
    if output_path.exists():
        return str(output_path)

    try:
        parsed_data = svg_parser.parse_svg(svg_file_path)
        viewinfo = parsed_data['viewport']
        elements_from_parser = parsed_data.get('elements', [])
        if not elements_from_parser: return None
    except Exception as e:
        print(f" Error parsing {file_name_stem}: {e}")
        traceback.print_exc()
        return None

    # 1. Get the full SVG matrix content (unpadded, normalized)
    full_element_tensor_parts = []
    element_indices_that_produced_tensors = [] 

    for i, element_data in enumerate(elements_from_parser):
        tensor_for_current_el = svg_tensor_converter.create_tensor_for_element(element_data)
        if tensor_for_current_el is not None and tensor_for_current_el.shape[0] > 0:
            full_element_tensor_parts.append(tensor_for_current_el)
            element_indices_that_produced_tensors.append(i)

    if not full_element_tensor_parts: return None
    full_svg_matrix_content = torch.cat(full_element_tensor_parts, dim=0)
    
    actual_element_row_counts = torch.tensor([part.shape[0] for part in full_element_tensor_parts], dtype=torch.long)

    # 2. Get cumulative pixel patch tokens (sequences, not just CLS)
    cumulative_pixel_patch_tokens = []
    current_raster_element_strings = []
    last_successful_embedding_sequence = torch.zeros(fixed_dino_patch_seq_length, dino_embed_dim, dtype=torch.float32)

    for i_el_parser, element_data_for_raster in enumerate(elements_from_parser):
        element_k_string = parse_element_to_string(element_data_for_raster)
        if not element_k_string: 
            cumulative_pixel_patch_tokens.append(last_successful_embedding_sequence.clone())
            continue 
            
        current_raster_element_strings.append(element_k_string)
        svg_content_cumulative_k = _create_cumulative_svg_string_static(current_raster_element_strings, viewinfo)
        embedding_cumulative_k = _rasterize_and_embed_static(svg_content_cumulative_k, viewinfo, dino_model, dino_processor, dino_device)
        
        if embedding_cumulative_k is not None:
            if embedding_cumulative_k.shape[0] == fixed_dino_patch_seq_length:
                last_successful_embedding_sequence = embedding_cumulative_k
            else:
                print(f"  Warning: DINOv2 output seq length mismatch for {file_name_stem} at stage {i_el_parser}. Expected {fixed_dino_patch_seq_length}, got {embedding_cumulative_k.shape[0]}. Using last successful sequence.")
        cumulative_pixel_patch_tokens.append(last_successful_embedding_sequence.clone())
    
    if not cumulative_pixel_patch_tokens:
        return None

    # ALIGNMENT
    aligned_pixel_patch_tokens_for_tensor_stages = []
    for i_tensor_stage in range(len(actual_element_row_counts)):
        original_element_parser_idx = element_indices_that_produced_tensors[i_tensor_stage]
        if original_element_parser_idx < len(cumulative_pixel_patch_tokens):
            aligned_pixel_patch_tokens_for_tensor_stages.append(cumulative_pixel_patch_tokens[original_element_parser_idx])
        else:
            print(f"Warning: Alignment issue for {file_name_stem}. Missing pixel token sequence for tensor stage. Using last or zero.")
            aligned_pixel_patch_tokens_for_tensor_stages.append(
                cumulative_pixel_patch_tokens[-1] if cumulative_pixel_patch_tokens else torch.zeros(fixed_dino_patch_seq_length, dino_embed_dim, dtype=torch.float32)
            )
            
    if not aligned_pixel_patch_tokens_for_tensor_stages: return None

    # --- MODIFIED: Save the precomputed data to a file and return its path ---
    precomputed_data_for_svg = {
        'file_name_stem': file_name_stem,
        'full_svg_matrix_content': full_svg_matrix_content,
        'cumulative_pixel_patch_tokens_aligned': torch.stack(aligned_pixel_patch_tokens_for_tensor_stages),
        'element_row_counts_for_stages': actual_element_row_counts
    }
    torch.save(precomputed_data_for_svg, output_path)
    return str(output_path) # Return the path to the saved file


# --- PyTorch Dataset for Dynamic Progressive Build ---
class DynamicProgressiveSVGDataset(Dataset):
    def __init__(self, precomputed_file_paths, max_seq_length, 
                 sos_token_row, eos_token_row, padding_row_template,
                 fixed_dino_patch_seq_length
                ):
        self.precomputed_file_paths = precomputed_file_paths # Now a list of file paths
        self.max_seq_length = max_seq_length

        self.sos_token_row = sos_token_row
        self.eos_token_row = eos_token_row
        self.padding_row_template = padding_row_template
        self.fixed_dino_patch_seq_length = fixed_dino_patch_seq_length
        
        self.index_map = []
        self.num_total_progressive_samples = 0
        for orig_idx, file_path in enumerate(self.precomputed_file_paths): # Iterate over file paths
            # Load the item_data just to get num_stages for indexing.
            # This is a small overhead at init, but can be optimized if num_stages is predictable.
            # For now, loading the minimal info needed for index_map.
            # A more robust way might be to save (file_path, num_stages) tuples.
            temp_data = torch.load(file_path)
            num_stages_for_this_item = len(temp_data['element_row_counts_for_stages'])
            for stage_k in range(num_stages_for_this_item):
                self.index_map.append({'original_item_idx': orig_idx, 'stage_k': stage_k})
            self.num_total_progressive_samples += num_stages_for_this_item
        
        print(f"DynamicProgressiveSVGDataset: Loaded {len(self.precomputed_file_paths)} original SVGs, yielding {self.num_total_progressive_samples} progressive samples.")

    def __len__(self):
        return self.num_total_progressive_samples

    def __getitem__(self, global_idx):
        #print(f"Fetching item {global_idx}/{self.num_total_progressive_samples}")
        map_info = self.index_map[global_idx]
        original_item_idx = map_info['original_item_idx']
        stage_k = map_info['stage_k']

        # --- MODIFIED PART START: Load data on demand ---
        file_path = self.precomputed_file_paths[original_item_idx]
        #print(file_path)
        original_item_data = torch.load(file_path) # Load only the needed SVG's data
        # --- MODIFIED PART END ---

        full_svg_matrix_content = original_item_data['full_svg_matrix_content']
        pixel_kv_sequence = original_item_data['cumulative_pixel_patch_tokens_aligned'][stage_k, :, :] # Shape: [M, D_dino]
        element_row_counts = original_item_data['element_row_counts_for_stages']

        num_rows_for_stage_k = element_row_counts[:stage_k + 1].sum().item()
        svg_matrix_content_stage_k = full_svg_matrix_content[:num_rows_for_stage_k, :]

        svg_sequence_stage_k = torch.cat([
            self.sos_token_row.unsqueeze(0),
            svg_matrix_content_stage_k,
            self.eos_token_row.unsqueeze(0)
        ], dim=0)
        
        pixel_padded_kv = pixel_kv_sequence 

        current_svg_len = svg_sequence_stage_k.shape[0]
        svg_padded_q = svg_sequence_stage_k

        attention_mask_svg_q = torch.zeros(current_svg_len, dtype=torch.bool)

        if current_svg_len > self.max_seq_length:
            svg_padded_q = svg_sequence_stage_k[:self.max_seq_length, :]
            attention_mask_svg_q = torch.zeros(self.max_seq_length, dtype=torch.bool)
            if not torch.equal(svg_padded_q[-1, :3], self.eos_token_row[:3]): 
                 svg_padded_q[-1] = self.eos_token_row
        elif current_svg_len < self.max_seq_length:
            pad_len = self.max_seq_length - current_svg_len
            svg_padding_rows = self.padding_row_template.unsqueeze(0).repeat(pad_len, 1)
            mask_padding_rows = torch.ones(pad_len, dtype=torch.bool)
            svg_padded_q = torch.cat([svg_sequence_stage_k, svg_padding_rows], dim=0)
            attention_mask_svg_q = torch.cat([torch.zeros(current_svg_len, dtype=torch.bool), mask_padding_rows], dim=0)
        
        attention_mask_pixel_kv = torch.zeros(pixel_padded_kv.shape[0], dtype=torch.bool)

        return svg_padded_q, pixel_padded_kv, attention_mask_svg_q, attention_mask_pixel_kv


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    SVG_DIRECTORY = '/Users/varun_jagannath/Documents/D/svg_downloader/' # ADJUST
    SVG_PATTERN = "full_svg_clean_repo/twemoji*/*.svg"                 # ADJUST as needed for more files
    # SVG_PATTERN = "full_svg_clean_repo/twemoji*/umbrella.svg" # Smaller pattern for testing
    PRECOMPUTED_DATA_OUTPUT_DIR = "precomputed_patch_tokens_data" # New: Directory to save individual SVG data files
    PRECOMPUTED_FILE_LIST_PATH = "precomputed_patch_tokens_file_list.pt" # New: File to save the list of paths
    MAX_SEQ_LENGTH_DATASET = 1024 # Max length for padding in __getitem__ (for SVG queries)
    # --- END CONFIGURATION ---

    # Create the output directory if it doesn't exist
    Path(PRECOMPUTED_DATA_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Initialize components needed for preprocessing
    print("Initializing components for preprocessing...")
    svg_tensor_converter = SVGToTensor_Normalized()
    svg_parser = SVGParser()
    dino_model, dino_processor, dino_device, dino_embed_dim, fixed_dino_patch_seq_length = load_dino_model_components() 

    # 2. Preprocess all SVGs
    precomputed_file_paths = [] # Now stores paths to files
    svg_files = list(Path(SVG_DIRECTORY).glob(SVG_PATTERN))
    if not svg_files:
        print(f"No SVG files found for pattern '{SVG_PATTERN}' in '{SVG_DIRECTORY}'. Exiting.")
        sys.exit(1)
    print(f"Found {len(svg_files)} SVG files. Starting preprocessing...")

    for svg_file_path in tqdm(svg_files, desc="Preprocessing SVGs"):
        try:
            # Pass the output directory
            saved_file_path = preprocess_svg_for_dynamic_storage(
                str(svg_file_path), svg_tensor_converter, svg_parser,
                dino_model, dino_processor, dino_device, dino_embed_dim, fixed_dino_patch_seq_length,
                PRECOMPUTED_DATA_OUTPUT_DIR
            )
            if saved_file_path:
                precomputed_file_paths.append(saved_file_path)
        except Exception as e:
            print(f"Failed to process {svg_file_path}: {e}")
            traceback.print_exc()

    if not precomputed_file_paths:
        print("No data was successfully preprocessed. Exiting.")
        sys.exit(1)

    print(f"\nPreprocessing complete. Saving {len(precomputed_file_paths)} file paths to '{PRECOMPUTED_FILE_LIST_PATH}'...")
    torch.save(precomputed_file_paths, PRECOMPUTED_FILE_LIST_PATH)
    print("Precomputed file paths saved.")

    # 3. Instantiate and test DynamicProgressiveSVGDataset
    print("\nLoading precomputed file paths and instantiating DynamicProgressiveSVGDataset...")
    
    # --- Reconstruct SOS/EOS/PAD tokens based on the SVGToTensor_Normalized instance ---
    element_min = min(svg_tensor_converter.ELEMENT_TYPES.values())
    element_max = max(svg_tensor_converter.ELEMENT_TYPES.values())
    cmd_type_min = min(svg_tensor_converter.PATH_COMMAND_TYPES.values())
    cmd_type_max = max(svg_tensor_converter.PATH_COMMAND_TYPES.values())
    cmd_seq_idx_min = getattr(svg_tensor_converter, 'cmd_seq_idx_min', 0.0)
    cmd_seq_idx_max = getattr(svg_tensor_converter, 'cmd_seq_idx_max', 100.0)

    norm_sos_elem = svg_tensor_converter.ELEMENT_TYPES['<BOS>']
    norm_eos_elem = svg_tensor_converter.ELEMENT_TYPES['<EOS>']
    norm_pad_elem = svg_tensor_converter.ELEMENT_TYPES['<PAD>']
    
    norm_zero_cmd_seq = 0.0
    norm_no_cmd_type = svg_tensor_converter.PATH_COMMAND_TYPES['NO_CMD']

    default_params = torch.full((svg_tensor_converter.num_geom_params + svg_tensor_converter.num_fill_style_params,),
                                svg_tensor_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)

    sos_token_row = torch.cat([torch.tensor([norm_sos_elem, norm_zero_cmd_seq, norm_no_cmd_type], dtype=torch.float32), default_params])
    eos_token_row = torch.cat([torch.tensor([norm_eos_elem, norm_zero_cmd_seq, norm_no_cmd_type], dtype=torch.float32), default_params])
    padding_row_template = torch.cat([torch.tensor([norm_pad_elem, norm_zero_cmd_seq, norm_no_cmd_type], dtype=torch.float32), default_params])
    # --- End token reconstruction ---

    dynamic_dataset = DynamicProgressiveSVGDataset(
        precomputed_file_paths=torch.load(PRECOMPUTED_FILE_LIST_PATH), # Load the list of paths
        max_seq_length=MAX_SEQ_LENGTH_DATASET,
        sos_token_row=sos_token_row,
        eos_token_row=eos_token_row,
        padding_row_template=padding_row_template,
        fixed_dino_patch_seq_length=fixed_dino_patch_seq_length
    )

    if len(dynamic_dataset) == 0:
        print("Dynamic dataset is empty. Check preprocessing. Exiting.")
        sys.exit(1)

    print(f"\nTesting DataLoader with DynamicProgressiveSVGDataset (Total samples: {len(dynamic_dataset)})...")
    
    def simple_collate(batch):
        svg_matrices = torch.stack([item[0] for item in batch])
        pixel_embeds_kv = torch.stack([item[1] for item in batch])
        attention_masks_q = torch.stack([item[2] for item in batch])
        attention_masks_kv = torch.stack([item[3] for item in batch])
        return svg_matrices, pixel_embeds_kv, attention_masks_q, attention_masks_kv

    # Use num_workers > 0 for parallel loading from disk
    dataloader = DataLoader(dynamic_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=simple_collate) 

    try:
        for i, batch_data in enumerate(tqdm(dataloader, desc="DataLoader Test")):
            svg_matrices, pixel_embeddings, attention_mask_q, attention_mask_kv = batch_data
            print(f"\nBatch {i+1} Shapes:")
            print(f" SVG Queries (Q):     {svg_matrices.shape}") # Expect [B, MAX_SEQ_LENGTH_DATASET, N_Features]
            print(f" Pixel Keys/Values (KV): {pixel_embeddings.shape}") # Expect [B, M, dino_embed_dim]
            print(f" SVG Q Attention Mask: {attention_mask_q.shape}")  # Expect [B, MAX_SEQ_LENGTH_DATASET]
            print(f" Pixel KV Attention Mask:{attention_mask_kv.shape}") # Expect [B, M]

            if i == 0: 
                print("Example SVG Matrix (Batch 0, Item 0, Rows 0-5):")
                print(svg_matrices[0, :5, :])
                print("Example Pixel Embedding (Batch 0, Item 0, Rows 0-5, Cols 0-5):")
                print(pixel_embeddings[0, :5, :5])
                print("Example SVG Q Attention Mask (Batch 0, Item 0, Rows 0-5):")
                print(attention_mask_q[0, :5])
                print("Example Pixel KV Attention Mask (Batch 0, Item 0, Rows 0-5):")
                print(attention_mask_kv[0, :5])
            if i >= 1: 
               break
        print("\nDataLoader test successful with DynamicProgressiveSVGDataset (Patch Tokens).")
    except Exception as e:
         print(f"\nError during DataLoader test:")
         traceback.print_exc()
