# --- START OF FILE tensorsvg.py (with refined path handling) ---
import torch
import math
from pathlib import Path

class TensorToSVGHybrid:
    def __init__(self, svg_tensor_converter_instance):
        self.converter = svg_tensor_converter_instance
        self.element_map_rev = {v: k for k, v in self.converter.ELEMENT_TYPES.items()}
        self.command_map_rev = {v: k for k, v in self.converter.PATH_COMMAND_TYPES.items()}
        self.CMD_DEF_ID = self.converter.PATH_COMMAND_TYPES.get('DEF', -99)
        # Path command IDs for quick check
        self.PATH_CMD_IDS = {
            self.converter.PATH_COMMAND_TYPES.get(cmd_char.lower(), -100 - idx) # Ensure unique if not found
            for idx, cmd_char in enumerate("mlhvcsqtaz") # Common path commands
        }
        self.CMD_M_ID = self.converter.PATH_COMMAND_TYPES.get('m', -1)
        self.CMD_Z_ID = self.converter.PATH_COMMAND_TYPES.get('z', -1)


    def _denormalize(self, norm_value, val_min, val_max):
        # ... (denormalization logic as before) ...
        target_min = self.converter.target_norm_min
        target_max = self.converter.target_norm_max
        if target_max == target_min: return target_min
        if val_max == val_min: return val_min
        norm_0_1 = (norm_value - target_min) / (target_max - target_min)
        value = norm_0_1 * (val_max - val_min) + val_min
        return value

    def _format_geo_params_for_path_d(self, cmd_id, geo_params_norm):
        # ... (geo param formatting as before) ...
        params_str_list = []
        cmd_char = self.command_map_rev.get(cmd_id, '?').lower()
        #print(cmd_char)
        try:
            if cmd_char in ['m', 'l', 't']: 
                params_str_list.append(f"{self._denormalize(geo_params_norm[6].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[7].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            elif cmd_char == 'h': 
                params_str_list.append(f"{self._denormalize(geo_params_norm[6].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            elif cmd_char == 'v': 
                params_str_list.append(f"{self._denormalize(geo_params_norm[7].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            elif cmd_char == 'c': 
                for i in range(2, 8): 
                    params_str_list.append(f"{self._denormalize(geo_params_norm[i].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            elif cmd_char in ['q', 's']: 
                params_str_list.append(f"{self._denormalize(geo_params_norm[2].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[3].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[6].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[7].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            elif cmd_char == 'a': 
                params_str_list.append(f"{self._denormalize(geo_params_norm[0].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[1].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[2].item(), self.converter.ROT_MIN, self.converter.ROT_MAX):.1f}")    
                params_str_list.append(f"{int(round(self._denormalize(geo_params_norm[3].item(), self.converter.FLAG_MIN, self.converter.FLAG_MAX)))}")
                params_str_list.append(f"{int(round(self._denormalize(geo_params_norm[4].item(), self.converter.FLAG_MIN, self.converter.FLAG_MAX)))}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[5].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
                params_str_list.append(f"{self._denormalize(geo_params_norm[6].item(), self.converter.COORD_MIN, self.converter.COORD_MAX):.1f}")
            elif cmd_char == 'z': pass 
        except IndexError: print(f"WARN: IndexError formatting geo params for cmd '{cmd_char}'. Geo params: {geo_params_norm}")
        return " ".join(params_str_list)


    def _format_style_attributes(self, style_params_norm):
        # ... (style formatting as before) ...
        print(style_params_norm)
        attrs = []
        try:
            r_norm = style_params_norm[0].item() if torch.is_tensor(style_params_norm[0]) else style_params_norm[0]
            g_norm = style_params_norm[1].item() if torch.is_tensor(style_params_norm[1]) else style_params_norm[1]
            b_norm = style_params_norm[2].item() if torch.is_tensor(style_params_norm[2]) else style_params_norm[2]
            #a_norm = style_params_norm[3].item() if torch.is_tensor(style_params_norm[3]) else style_params_norm[3]

            r = int(round(self._denormalize(r_norm, self.converter.COLOR_MIN, self.converter.COLOR_MAX)))
            g = int(round(self._denormalize(g_norm, self.converter.COLOR_MIN, self.converter.COLOR_MAX)))
            b = int(round(self._denormalize(b_norm, self.converter.COLOR_MIN, self.converter.COLOR_MAX)))
            #alpha = self._denormalize(a_norm, self.converter.OPACITY_MIN, self.converter.OPACITY_MAX)

            r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
            alpha = 1.0
            print(r, g,b)

            attrs.append(f'fill="#{r:02x}{g:02x}{b:02x}"')
            if alpha < 0.995: attrs.append(f'fill-opacity="{alpha:.2f}"')
            attrs.append('stroke="none"')
        except IndexError:
            print(f"WARN: IndexError formatting style attributes. Style params: {style_params_norm}") 
            attrs.append('fill="black" stroke="none"')
        return " ".join(attrs)

    def reconstruct_svg_elements(self, tensor_data_hybrid, actual_len=None):
        if actual_len is not None:
            tensor_to_process = tensor_data_hybrid[:actual_len]
        else:
            tensor_to_process = tensor_data_hybrid
        
        svg_elements_strings = []
        
        # State for current path being built
        current_path_d_segments = []
        current_path_style_params = None # Store style params for the current path

        for i in range(tensor_to_process.shape[0]):
            row = tensor_to_process[i]
            elem_id = int(row[0].item())
            cmd_id = int(row[1].item())
            geo_params_norm = row[2 : 2+self.converter.num_geom_params] 
            style_params_norm = row[2+self.converter.num_geom_params : 2+self.converter.num_geom_params+self.converter.num_fill_style_params]

            elem_tag_str = self.element_map_rev.get(elem_id, None)
            #print(elem_tag_str)

            if elem_tag_str is None or elem_tag_str in ['<BOS>', '<PAD>']:
                continue
            if elem_tag_str == '<EOS>':
                break 

            is_current_row_a_path_command = (elem_tag_str == "path" and cmd_id in self.PATH_CMD_IDS)
            is_moveto_command = (cmd_id == self.CMD_M_ID)
            is_closepath_command = (cmd_id == self.CMD_Z_ID)

            # --- Logic to finalize a path ---
            # Finalize if:
            # 1. We are in a path AND current row is NOT a path command (it's a new element type)
            # 2. We are in a path AND current row IS an 'M' command (signaling a new subpath that we'll treat as a new <path> element for simplicity)
            should_finalize_path = current_path_d_segments and (not is_current_row_a_path_command or is_moveto_command)

            if should_finalize_path:
                if current_path_style_params is not None:
                    style_attrs_str = self._format_style_attributes(current_path_style_params)
                    svg_elements_strings.append(f'  <path d="{" ".join(current_path_d_segments)}" {style_attrs_str}/>')
                else: # Fallback style if none was captured (should not happen if M command always has style)
                    svg_elements_strings.append(f'  <path d="{" ".join(current_path_d_segments)}" fill="#777777" stroke="none"/> <!-- Default style for path -->')
                current_path_d_segments = [] # Reset for a potentially new path
                current_path_style_params = None

            # --- Process current row ---
            if is_current_row_a_path_command:
                if not current_path_d_segments or is_moveto_command: # Start of a new path or forced new path by M
                    current_path_d_segments = [] # Ensure it's empty for a new M
                    current_path_style_params = style_params_norm # Capture style at the start of the path segment (M)
                
                # For subsequent commands in the same path, we could choose to update style or keep the first one.
                # Here, we update `current_path_style_params` with each command's style, effectively using the last one.
                # A better approach might be to use the style from the M command.
                # For now, let's use the style from the *current* path command row for `last_valid_style_params_for_path`
                current_path_style_params = style_params_norm # This means style can change mid-path if model predicts it

                path_geo_str = self._format_geo_params_for_path_d(cmd_id, geo_params_norm)
                cmd_char_for_d = self.command_map_rev.get(cmd_id, '?') # M, L, Z etc.
                current_path_d_segments.append(f"{cmd_char_for_d} {path_geo_str}".strip())

                if is_closepath_command and current_path_d_segments: # Finalize on Z
                    if current_path_style_params is not None:
                         style_attrs_str = self._format_style_attributes(current_path_style_params)
                         svg_elements_strings.append(f'  <path d="{" ".join(current_path_d_segments)}" {style_attrs_str}/>')
                    else:
                         svg_elements_strings.append(f'  <path d="{" ".join(current_path_d_segments)}" fill="#555555" stroke="none"/> <!-- Default style for Z-closed path -->')
                    current_path_d_segments = []
                    current_path_style_params = None
            
            elif elem_tag_str == "rect" and cmd_id == self.CMD_DEF_ID:
                x = self._denormalize(geo_params_norm[0].item(), self.converter.COORD_MIN, self.converter.COORD_MAX)
                y = self._denormalize(geo_params_norm[1].item(), self.converter.COORD_MIN, self.converter.COORD_MAX)
                rx = self._denormalize(geo_params_norm[2].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                ry = self._denormalize(geo_params_norm[3].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                w = self._denormalize(geo_params_norm[4].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                h = self._denormalize(geo_params_norm[5].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                style_attrs_str = self._format_style_attributes(style_params_norm)
                svg_elements_strings.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" {style_attrs_str}/>')

            elif elem_tag_str == "circle" and cmd_id == self.CMD_DEF_ID:
                cx = self._denormalize(geo_params_norm[0].item(), self.converter.COORD_MIN, self.converter.COORD_MAX)
                cy = self._denormalize(geo_params_norm[1].item(), self.converter.COORD_MIN, self.converter.COORD_MAX)
                r  = self._denormalize(geo_params_norm[2].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                style_attrs_str = self._format_style_attributes(style_params_norm)
                svg_elements_strings.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {style_attrs_str}/>')
            
            elif elem_tag_str == "ellipse" and cmd_id == self.CMD_DEF_ID:
                cx = self._denormalize(geo_params_norm[0].item(), self.converter.COORD_MIN, self.converter.COORD_MAX)
                cy = self._denormalize(geo_params_norm[1].item(), self.converter.COORD_MIN, self.converter.COORD_MAX)
                rx = self._denormalize(geo_params_norm[2].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                ry = self._denormalize(geo_params_norm[3].item(), self.converter.RADIUS_MIN, self.converter.RADIUS_MAX)
                style_attrs_str = self._format_style_attributes(style_params_norm)
                svg_elements_strings.append(f'  <ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" {style_attrs_str}/>')
            
            # else: unknown element/command combo, or already handled path segment

        # Finalize any pending path after loop
        if current_path_d_segments:
            if current_path_style_params is not None:
                style_attrs_str = self._format_style_attributes(current_path_style_params)
                svg_elements_strings.append(f'  <path d="{" ".join(current_path_d_segments)}" {style_attrs_str}/>')
            else: # Should have style if segments exist
                 svg_elements_strings.append(f'  <path d="{" ".join(current_path_d_segments)}" fill="#333333" stroke="none"/> <!-- Default style for trailing path -->')
        
        return svg_elements_strings

    def create_svg_document(self, svg_element_strings_list, viewbox_dims=(128,128)):
        # ... (as before) ...
        width, height = viewbox_dims
        viewBox_str = f"0 0 {width} {height}"
        svg_header = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox_str}">'
        svg_footer = '</svg>'
        return f"{svg_header}\n" + "\n".join(svg_element_strings_list) + f"\n{svg_footer}"

# --- Wrapper Function (same as before) ---
def tensor_to_svg_file_hybrid_wrapper(
    predicted_hybrid_tensor_cpu, output_filename,
    svg_tensor_converter_instance, actual_len=None):
    # ... (as before) ...
    reconstructor = TensorToSVGHybrid(svg_tensor_converter_instance)
    svg_element_strings = reconstructor.reconstruct_svg_elements(predicted_hybrid_tensor_cpu, actual_len)
    viewbox_w = getattr(svg_tensor_converter_instance, 'viewBox_width_unnormalized', 128) 
    viewbox_h = getattr(svg_tensor_converter_instance, 'viewBox_height_unnormalized', 150) # Match test
    final_svg_content = reconstructor.create_svg_document(svg_element_strings, viewbox_dims=(int(viewbox_w), int(viewbox_h)))
    try:
        with open(output_filename, 'w') as f: f.write(final_svg_content)
        print(f"INFO: Reconstructed SVG saved to: {output_filename}")
    except Exception as e_write: print(f"ERROR: Failed to write SVG file {output_filename}: {e_write}")
    return final_svg_content

# --- __main__ for testing TensorToSVGHybrid (as before) ---
if __name__ == "__main__":
    # ... (your existing __main__ test block for TensorToSVGHybrid) ...
    print("Testing TensorToSVGHybrid with refined path handling...")
    
    class DummySVGToTensorForHybrid(SVGToTensor_Normalized):
        def __init__(self):
            super().__init__() 
            self.ELEMENT_TYPES = {'<BOS>': 0, '<EOS>': 1, '<PAD>': 2, '<rect>': 3, '<circle>': 4, '<path>': 5}
            # Simplified, ensure your actual PATH_COMMAND_TYPES is comprehensive
            self.PATH_COMMAND_TYPES = {'NO_CMD':0, 'DEF':1, 'm':2, 'l':3, 'c':4, 'z':5, 'a':6, 'v':7, 'h':8, 'q':9, 's':10, 't':11} 
            self.num_geom_params = 8 
            self.num_fill_style_params = 4 
            self.target_norm_min = -1.0; self.target_norm_max = 1.0
            self.COORD_MIN, self.COORD_MAX = -256.0, 255.0 
            self.RADIUS_MIN, self.RADIUS_MAX = 0.0, 255.0   
            self.ROT_MIN, self.ROT_MAX = -180.0, 180.0
            self.FLAG_MIN, self.FLAG_MAX = 0.0, 1.0
            self.COLOR_MIN, self.COLOR_MAX = 0.0, 255.0
            self.OPACITY_MIN, self.OPACITY_MAX = 0.0, 1.0
            self.DEFAULT_PARAM_VAL = 0.0 
            self.viewBox_width_unnormalized = 128; self.viewBox_height_unnormalized = 128

    dummy_converter_hybrid = DummySVGToTensorForHybrid()
    
    # Test tensor based on your example output that produced the single path
    # Using only path commands now to test grouping
    test_hybrid_tensor_path = torch.tensor([
        # ElemID, CmdID, geo0-7 (norm), R,G,B,A (norm)
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<path>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['m']),
         0,0,0,0,0,0, dummy_converter_hybrid._normalize(70.2, dummy_converter_hybrid.COORD_MIN, dummy_converter_hybrid.COORD_MAX), dummy_converter_hybrid._normalize(77.8, dummy_converter_hybrid.COORD_MIN, dummy_converter_hybrid.COORD_MAX),
         dummy_converter_hybrid._normalize(111,0,255), dummy_converter_hybrid._normalize(103,0,255), dummy_converter_hybrid._normalize(120,0,255), 1.0],
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<path>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['a']),
         dummy_converter_hybrid._normalize(6.2,0,255), dummy_converter_hybrid._normalize(5.8,0,255), dummy_converter_hybrid._normalize(-6.0,-180,180), dummy_converter_hybrid._normalize(0,0,1), dummy_converter_hybrid._normalize(1,0,1), dummy_converter_hybrid._normalize(41.8,-256,255), dummy_converter_hybrid._normalize(67.3,-256,255), 0.0, # unused geo
         dummy_converter_hybrid._normalize(115,0,255), dummy_converter_hybrid._normalize(107,0,255), dummy_converter_hybrid._normalize(131,0,255), 1.0],
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<path>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['v']),
         0,0,0,0,0,0, 0.0, dummy_converter_hybrid._normalize(55.2, dummy_converter_hybrid.COORD_MIN, dummy_converter_hybrid.COORD_MAX), # V uses only vy (ν3) -> geo_params[7]
         dummy_converter_hybrid._normalize(132,0,255), dummy_converter_hybrid._normalize(115,0,255), dummy_converter_hybrid._normalize(131,0,255), 1.0],
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<path>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['z']), # Z command
         0,0,0,0,0,0,0,0,
         dummy_converter_hybrid._normalize(112,0,255), dummy_converter_hybrid._normalize(85,0,255), dummy_converter_hybrid._normalize(107,0,255), 1.0],
        # Start a new path
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<path>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['m']),
         0,0,0,0,0,0, dummy_converter_hybrid._normalize(68.7,-256,255), dummy_converter_hybrid._normalize(54.4,-256,255),
         dummy_converter_hybrid._normalize(100,0,255), dummy_converter_hybrid._normalize(100,0,255), dummy_converter_hybrid._normalize(100,0,255), 1.0],
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<path>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['c']),
         0,0, dummy_converter_hybrid._normalize(59.7,-256,255), dummy_converter_hybrid._normalize(48.9,-256,255), dummy_converter_hybrid._normalize(56.5,-256,255), dummy_converter_hybrid._normalize(47.6,-256,255), dummy_converter_hybrid._normalize(59.3,-256,255), dummy_converter_hybrid._normalize(59.0,-256,255),
         dummy_converter_hybrid._normalize(100,0,255), dummy_converter_hybrid._normalize(100,0,255), dummy_converter_hybrid._normalize(100,0,255), 1.0],
        [float(dummy_converter_hybrid.ELEMENT_TYPES['<EOS>']), float(dummy_converter_hybrid.PATH_COMMAND_TYPES['NO_CMD']),
         0,0,0,0,0,0,0,0, 0,0,0,0]
    ], dtype=torch.float32)

    output_svg_path_new = "test_reconstruction_hybrid_grouped_paths.svg"
    tensor_to_svg_file_hybrid_wrapper(
        test_hybrid_tensor_path,
        output_svg_path_new,
        dummy_converter_hybrid, 
        actual_len=6 # Process up to the second C command
    )
    print(f"Test SVG with grouped paths saved to {output_svg_path_new}.")
# --- END OF FILE tensorsvg.py ---
