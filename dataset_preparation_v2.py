# --- START OF FILE dataset_preparation_dynamic.py ---

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
# import torch.nn.functional as F # Not directly used in this script's main logic
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import traceback
from io import BytesIO
import cairosvg
import numpy as np
import os
from pathlib import Path

from svgutils import SVGToTensor_Normalized # MUST be the version outputting fully normalized floats
from svgutils import SVGParser

# --- DINOv2 Loading Utility ---
def load_dino_model_components():
    model_name = 'facebook/dinov2-small'
    print(f"Loading DINOv2 components: {model_name}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # MPS can be slow for DINO
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
        print(f"DINOv2 components loaded successfully. Using device: {device}")
        return model, processor, device, model.config.hidden_size
    except Exception as e:
        print(f"Error loading DINOv2 components: {e}"); exit()

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
    #print(svg_content_string)
    try:
        output_w = max(1, int(float(viewinfo.get('width', 512))))
        output_h = max(1, int(float(viewinfo.get('height', 512))))
        png_buffer = BytesIO()
        cairosvg.svg2png(bytestring=svg_content_string.encode('utf-8'), output_width=output_w, output_height=output_h, write_to=png_buffer)
        png_buffer.seek(0)
        image = Image.open(png_buffer).convert("RGB")
    except Exception as e: print(f"  Error rasterizing SVG: {e}"); return None
    try:
        with torch.no_grad():
            inputs = dino_processor(images=image, return_tensors="pt").to(device)
            outputs = dino_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0] # CLS token
            #print(embedding.squeeze(0).cpu())
            return embedding.squeeze(0).cpu()
    except Exception as e: print(f"  Error getting DINO embedding: {e}"); return None

def parse_element_to_string(element_data): # Simplified combined parser
    #print(element_data)
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
            if len(parts) > 5 and ("a" in parts[0].lower() or "A" in parts[0].lower()): # check lower for safety
                try: 
                    parts[2] = str(int(float(parts[2]))); parts[3] = str(int(float(parts[3]))); parts[4] = str(int(float(parts[4])))
                except ValueError: print(f"  Warning: Could not parse arc flags/rot in {parts}")
            d_parts.append(" ".join(parts))
        path_str += " ".join(d_parts) + '"' + style_string + '/>'
        return path_str
    
    elif el_type == 'circle':
        #print(element_data)
        #attrs = element_data.get('commands', {})
        attrs = {item['command']: item['values'] for item in element_data.get('commands', {})}
        #print(attrs)
        cx = attrs.get('cx', 0); cy = attrs.get('cy', 0); r = attrs.get('r', 0)
        return f'<circle cx="{cx}" cy="{cy}" r="{r}"{style_string}/>'
    
    elif el_type == 'rect':
        #attrs = element_data.get('commands', {})
        attrs = {item['command']: item['values'] for item in element_data.get('commands', {})}
        attr_str_parts = [f'{k}="{v}"' for k, v in attrs.items()]
        return f'<rect {" ".join(attr_str_parts)}{style_string}/>'

    elif el_type == 'ellipse':
        #attrs = element_data.get('commands', {})
        attrs = {item['command']: item['values'] for item in element_data.get('commands', {})}
        cx = attrs.get('cx',0); cy = attrs.get('cy',0); rx = attrs.get('rx',0); ry = attrs.get('ry',0)
        return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"{style_string}/>'
    else:
        return ""


# --- Preprocessing Function for a Single SVG ---
def preprocess_svg_for_dynamic_storage(
    svg_file_path, svg_tensor_converter, svg_parser,
    dino_model, dino_processor, dino_device, dino_embed_dim
):
    file_name_stem = Path(svg_file_path).stem
    try:
        parsed_data = svg_parser.parse_svg(svg_file_path)
        viewinfo = parsed_data['viewport']
        elements_from_parser = parsed_data.get('elements', [])
        if not elements_from_parser: return None
    except Exception as e:
        print(f" Error parsing {file_name_stem}: {e}"); return None

    # 1. Get the full SVG matrix content (unpadded, normalized)
    full_element_tensor_parts = []
    # Defines which original elements contributed tensor rows and how many.
    # This is key for aligning with pixel embeddings if an "element" for pixel
    # is one of the original elements_from_parser.
    element_indices_that_produced_tensors = [] 

    for i, element_data in enumerate(elements_from_parser):
        tensor_for_current_el = svg_tensor_converter.create_tensor_for_element(element_data)
        if tensor_for_current_el is not None and tensor_for_current_el.shape[0] > 0:
            full_element_tensor_parts.append(tensor_for_current_el)
            element_indices_that_produced_tensors.append(i) # Store original index

    if not full_element_tensor_parts: return None
    full_svg_matrix_content = torch.cat(full_element_tensor_parts, dim=0)
    
    # Derive element_row_counts for the tensor parts we actually stored
    actual_element_row_counts = torch.tensor([part.shape[0] for part in full_element_tensor_parts], dtype=torch.long)

    # 2. Get cumulative pixel CLS tokens
    # Pixel embeddings are generated for *every* original visual element,
    # even if it didn't produce tensor rows (e.g., an unknown type we skip in tensorization but still draw).
    cumulative_pixel_CLS_tokens = []
    current_raster_element_strings = [] # Strings for rasterizer
    last_successful_embedding = torch.zeros(dino_embed_dim, dtype=torch.float32)

    for i_el_parser, element_data_for_raster in enumerate(elements_from_parser):
        element_k_string = parse_element_to_string(element_data_for_raster)
        if not element_k_string: # If element is not drawable or unknown by parse_element_to_string
            # If this element was supposed to be a stage that contributes to the tensor,
            # we must have a pixel embedding. Use the last one.
            # Otherwise, if it's just an intermediate visual element with no tensor part,
            # we can decide if we need its specific embedding or if last_successful is fine.
            # For simplicity, we generate an embedding for each visual step.
            if cumulative_pixel_CLS_tokens: # check if list is not empty
                 cumulative_pixel_CLS_tokens.append(last_successful_embedding.clone())
            else: # First element failed to parse to string or was empty
                 cumulative_pixel_CLS_tokens.append(last_successful_embedding.clone()) # store zero embedding
            continue 
            
        current_raster_element_strings.append(element_k_string)
        svg_content_cumulative_k = _create_cumulative_svg_string_static(current_raster_element_strings, viewinfo)
        embedding_cumulative_k = _rasterize_and_embed_static(svg_content_cumulative_k, viewinfo, dino_model, dino_processor, dino_device)
        
        if embedding_cumulative_k is not None:
            last_successful_embedding = embedding_cumulative_k
        cumulative_pixel_CLS_tokens.append(last_successful_embedding.clone())
    
    if not cumulative_pixel_CLS_tokens: # Should not happen if elements_from_parser is not empty
        return None

    # ALIGNMENT: We need one pixel embedding per "stage" defined by actual_element_row_counts.
    # The i-th entry in actual_element_row_counts corresponds to the i-th part in full_svg_matrix_content.
    # This tensor part came from elements_from_parser[element_indices_that_produced_tensors[i]].
    # So, the pixel embedding for this stage should be cumulative_pixel_CLS_tokens[element_indices_that_produced_tensors[i]].
    
    aligned_pixel_CLS_for_tensor_stages = []
    for i_tensor_stage in range(len(actual_element_row_counts)):
        original_element_parser_idx = element_indices_that_produced_tensors[i_tensor_stage]
        if original_element_parser_idx < len(cumulative_pixel_CLS_tokens):
            aligned_pixel_CLS_for_tensor_stages.append(cumulative_pixel_CLS_tokens[original_element_parser_idx])
        else: # Should not happen if logic is correct
            print(f"Warning: Alignment issue for {file_name_stem}. Missing pixel token for tensor stage. Using last.")
            aligned_pixel_CLS_for_tensor_stages.append(cumulative_pixel_CLS_tokens[-1] if cumulative_pixel_CLS_tokens else torch.zeros(dino_embed_dim, dtype=torch.float32))
            
    if not aligned_pixel_CLS_for_tensor_stages: return None # No aligned pixel data

    return {
        'file_name_stem': file_name_stem,
        'full_svg_matrix_content': full_svg_matrix_content,
        'cumulative_pixel_CLS_tokens_aligned': torch.stack(aligned_pixel_CLS_for_tensor_stages), # Aligned with tensor stages
        'element_row_counts_for_stages': actual_element_row_counts
    }


# --- PyTorch Dataset for Dynamic Progressive Build ---
class DynamicProgressiveSVGDataset(Dataset):
    def __init__(self, precomputed_data_list, max_seq_length, 
                 sos_token_row, eos_token_row, padding_row_template, 
                 sos_pixel_embed, pixel_padding_template):
        self.precomputed_items = precomputed_data_list
        self.max_seq_length = max_seq_length

        self.sos_token_row = sos_token_row
        self.eos_token_row = eos_token_row
        self.padding_row_template = padding_row_template
        self.sos_pixel_embed = sos_pixel_embed # Should be [1, dino_embed_dim]
        self.pixel_padding_template = pixel_padding_template # Should be [1, dino_embed_dim]
        
        self.index_map = []
        self.num_total_progressive_samples = 0
        for orig_idx, item_data in enumerate(self.precomputed_items):
            # num_stages is defined by how many entries are in element_row_counts (i.e., how many tensor parts)
            num_stages_for_this_item = len(item_data['element_row_counts_for_stages'])
            for stage_k in range(num_stages_for_this_item): # 0-indexed stage_k
                self.index_map.append({'original_item_idx': orig_idx, 'stage_k': stage_k})
            self.num_total_progressive_samples += num_stages_for_this_item
        
        print(f"DynamicProgressiveSVGDataset: Loaded {len(self.precomputed_items)} original SVGs, yielding {self.num_total_progressive_samples} progressive samples.")

    def __len__(self):
        return self.num_total_progressive_samples

    def __getitem__(self, global_idx):
        map_info = self.index_map[global_idx]
        original_item_idx = map_info['original_item_idx']
        stage_k = map_info['stage_k'] # 0-indexed stage

        original_item_data = self.precomputed_items[original_item_idx]
        full_svg_matrix_content = original_item_data['full_svg_matrix_content']
        # Use the aligned pixel tokens
        pixel_CLS_tokens_for_tensor_stages = original_item_data['cumulative_pixel_CLS_tokens_aligned']
        element_row_counts = original_item_data['element_row_counts_for_stages']

        num_rows_for_stage_k = element_row_counts[:stage_k + 1].sum().item()
        svg_matrix_content_stage_k = full_svg_matrix_content[:num_rows_for_stage_k, :]

        svg_sequence_stage_k = torch.cat([
            self.sos_token_row.unsqueeze(0),
            svg_matrix_content_stage_k,
            self.eos_token_row.unsqueeze(0)
        ], dim=0)
        
        pixel_CLS_for_this_stage = pixel_CLS_tokens_for_tensor_stages[stage_k, :] # [D_dino]
        pixel_embed_content_stage_k = pixel_CLS_for_this_stage.unsqueeze(0).repeat(svg_matrix_content_stage_k.shape[0], 1)
        eos_pixel_stage_k = pixel_CLS_for_this_stage.unsqueeze(0)

        pixel_sequence_stage_k = torch.cat([
            self.sos_pixel_embed,
            pixel_embed_content_stage_k,
            eos_pixel_stage_k
        ], dim=0)

        current_len = svg_sequence_stage_k.shape[0]
        svg_padded = svg_sequence_stage_k
        pixel_padded = pixel_sequence_stage_k
        attention_mask = torch.zeros(current_len, dtype=torch.bool)

        if current_len > self.max_seq_length:
            svg_padded = svg_sequence_stage_k[:self.max_seq_length, :]
            pixel_padded = pixel_sequence_stage_k[:self.max_seq_length, :]
            attention_mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
            # Compare only a few key elements of EOS token if they are complex normalized floats
            # For simplicity, direct comparison. If using normalized floats, might need torch.allclose
            if not torch.equal(svg_padded[-1, :3], self.eos_token_row[:3]): 
                 svg_padded[-1] = self.eos_token_row
                 pixel_padded[-1] = eos_pixel_stage_k.squeeze(0)
        elif current_len < self.max_seq_length:
            pad_len = self.max_seq_length - current_len
            svg_padding_rows = self.padding_row_template.unsqueeze(0).repeat(pad_len, 1)
            pixel_padding_rows = self.pixel_padding_template.repeat(pad_len, 1)
            mask_padding_rows = torch.ones(pad_len, dtype=torch.bool)
            svg_padded = torch.cat([svg_sequence_stage_k, svg_padding_rows], dim=0)
            pixel_padded = torch.cat([pixel_sequence_stage_k, pixel_padding_rows], dim=0)
            attention_mask = torch.cat([torch.zeros(current_len, dtype=torch.bool), mask_padding_rows], dim=0)
        
        return svg_padded, pixel_padded, attention_mask


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    SVG_DIRECTORY = '/Users/varun_jagannath/Documents/D/svg_downloader/' # ADJUST
    # SVG_PATTERN = "full_svg_clean_repo/twemoji*/*.svg"                 # ADJUST
    SVG_PATTERN = "full_svg_clean_repo/twemoji*/*.svg" # Smaller pattern for testing
    PRECOMPUTED_OUTPUT_FILE = "optimized_progressive_dataset_precomputed_v2.pt"
    MAX_SEQ_LENGTH_DATASET = 1024 # Max length for padding in __getitem__
    # --- END CONFIGURATION ---

    # 1. Initialize components needed for preprocessing
    print("Initializing components for preprocessing...")
    svg_tensor_converter = SVGToTensor_Normalized() # ENSURE THIS IS THE FULLY NORMALIZED VERSION
    svg_parser = SVGParser()
    dino_model, dino_processor, dino_device, dino_embed_dim = load_dino_model_components()

    # 2. Preprocess all SVGs
    precomputed_data_list = []
    svg_files = list(Path(SVG_DIRECTORY).glob(SVG_PATTERN))
    if not svg_files:
        print(f"No SVG files found for pattern '{SVG_PATTERN}' in '{SVG_DIRECTORY}'. Exiting.")
        exit()
    print(f"Found {len(svg_files)} SVG files. Starting preprocessing...")

    for svg_file_path in tqdm(svg_files, desc="Preprocessing SVGs"):
        try:
            data_for_svg = preprocess_svg_for_dynamic_storage(
                str(svg_file_path), svg_tensor_converter, svg_parser,
                dino_model, dino_processor, dino_device, dino_embed_dim
            )
            if data_for_svg:
                precomputed_data_list.append(data_for_svg)
        except Exception as e:
            print(f"Failed to process {svg_file_path}: {e}")
            traceback.print_exc()

    if not precomputed_data_list:
        print("No data was successfully preprocessed. Exiting.")
        exit()

    print(f"\nPreprocessing complete. Saving {len(precomputed_data_list)} processed SVG items to '{PRECOMPUTED_OUTPUT_FILE}'...")
    torch.save(precomputed_data_list, PRECOMPUTED_OUTPUT_FILE)
    print("Precomputed data saved.")

    # 3. Instantiate and test DynamicProgressiveSVGDataset
    print("\nLoading precomputed data and instantiating DynamicProgressiveSVGDataset...")
    
    # Define SOS/EOS/Padding rows (MUST match the normalization used by SVGToTensor_Normalized)
    # These should come from or be consistent with svg_tensor_converter
    # For this example, let's get them based on SVGToTensor_Normalized's internal logic
    # This assumes SVGToTensor_Normalized has methods/attributes to get these normalized values
    # If not, they need to be hardcoded consistently or derived as in SVGPathRasterizerAligned
    
    # --- Reconstruct SOS/EOS/PAD tokens based on the SVGToTensor_Normalized instance ---
    # This part is crucial and must align with how SVGToTensor_Normalized outputs its data
    # (i.e., fully normalized floats for all 15 columns, including the first 3)
    element_min = min(svg_tensor_converter.ELEMENT_TYPES.values())
    element_max = max(svg_tensor_converter.ELEMENT_TYPES.values())
    cmd_type_min = min(svg_tensor_converter.PATH_COMMAND_TYPES.values()) # For cmd_type_idx
    cmd_type_max = max(svg_tensor_converter.PATH_COMMAND_TYPES.values())
    cmd_seq_idx_min = getattr(svg_tensor_converter, 'cmd_seq_idx_min', 0.0) # Tau min
    cmd_seq_idx_max = getattr(svg_tensor_converter, 'cmd_seq_idx_max', 100.0) # Tau max (example)

    # Assuming svg_tensor_converter._normalize exists and is the one used internally
    #norm_sos_elem = svg_tensor_converter._normalize(svg_tensor_converter.ELEMENT_TYPES['<BOS>'], element_min, element_max)
    norm_sos_elem = svg_tensor_converter.ELEMENT_TYPES['<BOS>']
    #norm_eos_elem = svg_tensor_converter._normalize(svg_tensor_converter.ELEMENT_TYPES['<EOS>'], element_min, element_max)
    norm_eos_elem = svg_tensor_converter.ELEMENT_TYPES['<EOS>']
    #norm_pad_elem = svg_tensor_converter._normalize(svg_tensor_converter.ELEMENT_TYPES['<PAD>'], element_min, element_max)
    norm_pad_elem = svg_tensor_converter.ELEMENT_TYPES['<PAD>']
    
    # cmd_seq_idx (tau) for SOS/EOS/PAD is 0, normalize it
    #norm_zero_cmd_seq = svg_tensor_converter._normalize(0.0, cmd_seq_idx_min, cmd_seq_idx_max)
    norm_zero_cmd_seq = 0.0
    
    # cmd_type_idx for SOS/EOS/PAD is NO_CMD, normalize it
    #norm_no_cmd_type = svg_tensor_converter._normalize(svg_tensor_converter.PATH_COMMAND_TYPES['NO_CMD'], cmd_type_min, cmd_type_max)
    norm_no_cmd_type = svg_tensor_converter.PATH_COMMAND_TYPES['NO_CMD']

    default_params = torch.full((svg_tensor_converter.num_geom_params + svg_tensor_converter.num_fill_style_params,),
                                svg_tensor_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)

    sos_token_row = torch.cat([torch.tensor([norm_sos_elem, norm_zero_cmd_seq, norm_no_cmd_type], dtype=torch.float32), default_params])
    eos_token_row = torch.cat([torch.tensor([norm_eos_elem, norm_zero_cmd_seq, norm_no_cmd_type], dtype=torch.float32), default_params])
    padding_row_template = torch.cat([torch.tensor([norm_pad_elem, norm_zero_cmd_seq, norm_no_cmd_type], dtype=torch.float32), default_params])
    
    sos_pixel_embed = torch.zeros((1, dino_embed_dim), dtype=torch.float32)
    pixel_padding_template = torch.zeros((1, dino_embed_dim), dtype=torch.float32)
    # --- End token reconstruction ---

    dynamic_dataset = DynamicProgressiveSVGDataset(
        precomputed_data_list=torch.load(PRECOMPUTED_OUTPUT_FILE), # Load the list directly
        max_seq_length=MAX_SEQ_LENGTH_DATASET,
        sos_token_row=sos_token_row,
        eos_token_row=eos_token_row,
        padding_row_template=padding_row_template,
        sos_pixel_embed=sos_pixel_embed,
        pixel_padding_template=pixel_padding_template
    )

    if len(dynamic_dataset) == 0:
        print("Dynamic dataset is empty. Check preprocessing. Exiting.")
        exit()

    print(f"\nTesting DataLoader with DynamicProgressiveSVGDataset (Total samples: {len(dynamic_dataset)})...")
    # Define a simple collate function if needed (usually not if __getitem__ returns tensors)
    def simple_collate(batch):
        svg_matrices = torch.stack([item[0] for item in batch])
        pixel_embeds = torch.stack([item[1] for item in batch])
        attention_masks = torch.stack([item[2] for item in batch])
        return svg_matrices, pixel_embeds, attention_masks

    dataloader = DataLoader(dynamic_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=simple_collate)

    try:
        for i, batch_data in enumerate(tqdm(dataloader, desc="DataLoader Test")):
            svg_matrices, pixel_embeddings, attention_mask = batch_data
            print(f"\nBatch {i+1} Shapes:")
            print(f" SVG Matrices:  {svg_matrices.shape}") # Expect [B, MAX_SEQ_LENGTH_DATASET, N_Features]
            print(f" Pixel Aligned: {pixel_embeddings.shape}")# Expect [B, MAX_SEQ_LENGTH_DATASET, dino_embed_dim]
            print(f" Attention Mask:{attention_mask.shape}")  # Expect [B, MAX_SEQ_LENGTH_DATASET]

            #print(svg_matrices[0, :50, :])
            #print(svg_matrices[1, :50, :])
            #print(svg_matrices[2, :50, :])
            #print(svg_matrices[3, :50, :])
            #print(pixel_embeddings[0, :50, :5])
            #print(pixel_embeddings[1, :50, :5])
            #print(pixel_embeddings[2, :50, :5])
            #print(pixel_embeddings[3, :50, :5])
            #print(attention_mask[0, :5])
            #print(attention_mask[1, :5])
            #print(attention_mask[2, :5])

            if i == 0: # Print details for the first batch only
                print("Example SVG Matrix (Batch 0, Item 0, Rows 0-5):")
                print(svg_matrices[0, :50, :])
                print("Example Pixel Embedding (Batch 0, Item 0, Rows 0-5, Cols 0-5):")
                print(pixel_embeddings[0, :50, :5])
                print("Example Attention Mask (Batch 0, Item 0, Rows 0-5):")
                print(attention_mask[0, :5])
            if i >= 1: # Test a couple of batches
               break
        print("\nDataLoader test successful with DynamicProgressiveSVGDataset.")
    except Exception as e:
         print(f"\nError during DataLoader test:")
         traceback.print_exc()

# --- END OF FILE dataset_preparation_dynamic.py ---
