# --- START OF FILE test30_v1_2_NORMALIZED.py ---

import torch
import math
from svgutils import SVGParser

# Assuming SVGParser is in test30.py
# from test30 import SVGParser

class SVGToTensor_Normalized_v0:
    def __init__(self, target_norm_min=-1.0, target_norm_max=1.0):
        # ρ (element type) + τ (command type) + 8 geometry + 9 style = max 19 features per row
        # We'll output ρ and τ as integers, and the rest as normalized floats.
        # The embedding layer in VP-VAE will handle ρ and τ.
        self.num_coord_params = 8
        self.num_style_params = 9 # Fill(R,G,B,A_fill), Stroke(R,G,B,A_stroke, Width)
        self.num_parameter_slots = max(self.num_coord_params, self.num_style_params) # 9

        # Total columns in the output matrix: ρ, τ, then N parameter slots
        self.output_matrix_cols = 1 + 1 + self.num_parameter_slots # e.g., 1+1+9 = 11 if only geometry/style in params

        # Parameter ranges (as before)
        self.COORD_MIN, self.COORD_MAX = -128.0, 127.0
        self.RADIUS_MIN, self.RADIUS_MAX = 0.0, 128.0
        self.FLAG_MIN, self.FLAG_MAX = 0.0, 1.0
        self.ROT_MIN, self.ROT_MAX = -180.0, 180.0
        self.OPACITY_MIN, self.OPACITY_MAX = 0.0, 1.0
        self.STROKEW_MIN, self.STROKEW_MAX = 0.0, 20.0
        self.COLOR_MIN, self.COLOR_MAX = 0.0, 255.0

        self.target_norm_min = target_norm_min
        self.target_norm_max = target_norm_max

        # Element type indices (ρ)
        self.ELEMENT_TYPES = {'<BOS>':0, 'rect': 1, 'circle': 2, 'ellipse': 3, 'path': 4, '<EOS>':5, '<PAD>':6 }
        self.PAD_ELEMENT_IDX = self.ELEMENT_TYPES['<PAD>']

        # Command type indices (τ)
        self.PATH_COMMAND_TYPES = {
            'm': 1, 'l': 2, 'h': 3, 'v': 4, 'c': 5, 's': 6, 'q': 7, 't': 8, 'a': 9, 'z': 10,
            'STYLE': 11, 'DEF': 12, 'NO_CMD': 0
        }
        self.NO_CMD_IDX = self.PATH_COMMAND_TYPES['NO_CMD']
        self.DEFAULT_PARAM_VAL = 0.0 # Default for unused normalized parameter slots


    def _normalize(self, value, val_min, val_max):
        """Normalizes value from [val_min, val_max] to [target_norm_min, target_norm_max]."""
        value = float(value) # Ensure float
        val_min, val_max = float(val_min), float(val_max)

        if val_max == val_min: # Avoid division by zero for constant features
            # Map to middle of target range or a fixed value like 0
            return (self.target_norm_min + self.target_norm_max) / 2.0

        # Clamp to original range first to handle outliers
        value_clamped = max(val_min, min(value, val_max))

        # Normalize to [0, 1]
        norm_0_1 = (value_clamped - val_min) / (val_max - val_min)
        # Scale to [target_norm_min, target_norm_max]
        return norm_0_1 * (self.target_norm_max - self.target_norm_min) + self.target_norm_min

    def _get_style_params_normalized(self, style_data):
        """Helper to normalize style parameters."""
        params_style = torch.full((self.num_style_params,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
        fill_rgb = style_data.get('fill_rgb', [0,0,0])
        stroke_rgb = style_data.get('stroke_rgb', [0,0,0])

        params_style[0] = self._normalize(fill_rgb[0], self.COLOR_MIN, self.COLOR_MAX)
        params_style[1] = self._normalize(fill_rgb[1], self.COLOR_MIN, self.COLOR_MAX)
        params_style[2] = self._normalize(fill_rgb[2], self.COLOR_MIN, self.COLOR_MAX)
        params_style[3] = self._normalize(style_data.get('opacity', 1.0), self.OPACITY_MIN, self.OPACITY_MAX) # fill-opacity
        params_style[4] = self._normalize(stroke_rgb[0], self.COLOR_MIN, self.COLOR_MAX)
        params_style[5] = self._normalize(stroke_rgb[1], self.COLOR_MIN, self.COLOR_MAX)
        params_style[6] = self._normalize(stroke_rgb[2], self.COLOR_MIN, self.COLOR_MAX)
        params_style[7] = self._normalize(style_data.get('stroke-opacity', 1.0), self.OPACITY_MIN, self.OPACITY_MAX)
        params_style[8] = self._normalize(style_data.get('stroke-width', 0.0), self.STROKEW_MIN, self.STROKEW_MAX)
        return params_style

    def create_tensor_for_element(self, element_data):
        element_type_str = element_data['type']
        commands_data = element_data.get('commands')
        style_data = element_data.get('style', {})

        element_rho = self.ELEMENT_TYPES.get(element_type_str, self.PAD_ELEMENT_IDX)
        if element_rho == self.PAD_ELEMENT_IDX and element_type_str not in ['<BOS>', '<EOS>', '<PAD>']:
            return None

        sequence_rows = []
        # Output tensor will have 2 integer columns (rho, tau) and then float params
        # So, create a float tensor and cast rho, tau later if needed by embedding layer,
        # or keep them as part of the float tensor if that's how VP-VAE consumes them.
        # For simplicity, let's make the whole row float and assume VP-VAE handles it.

        if element_type_str == 'path' and commands_data:
            current_x, current_y = 0.0, 0.0
            subpath_start_x, subpath_start_y = 0.0, 0.0

            for i, cmd_dict in enumerate(commands_data):
                row_params = torch.full((self.num_parameter_slots,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
                cmd_char = cmd_dict['command']
                cmd_tau = self.PATH_COMMAND_TYPES.get(cmd_char.lower(), self.NO_CMD_IDX)
                raw_values = cmd_dict.get('values', [])
                
                geom_params_normalized = torch.full((self.num_coord_params,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
                geom_params_normalized[0] = self._normalize(current_x, self.COORD_MIN, self.COORD_MAX)
                geom_params_normalized[1] = self._normalize(current_y, self.COORD_MIN, self.COORD_MAX)
                next_x, next_y = current_x, current_y

                if cmd_char.lower() == 'm':
                    if len(raw_values) >= 2:
                        next_x, next_y = raw_values[0], raw_values[1]
                        geom_params_normalized[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                        subpath_start_x, subpath_start_y = next_x, next_y
                elif cmd_char.lower() in ['l', 'h', 'v']:
                    if cmd_char.lower() == 'l' and len(raw_values) >= 2: next_x, next_y = raw_values[0], raw_values[1]
                    elif cmd_char.lower() == 'h' and len(raw_values) >= 1: next_x = raw_values[0]
                    elif cmd_char.lower() == 'v' and len(raw_values) >= 1: next_y = raw_values[0]
                    geom_params_normalized[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                    geom_params_normalized[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                elif cmd_char.lower() == 'c':
                    if len(raw_values) >= 6:
                        geom_params_normalized[2] = self._normalize(raw_values[0], self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[3] = self._normalize(raw_values[1], self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[4] = self._normalize(raw_values[2], self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[5] = self._normalize(raw_values[3], self.COORD_MIN, self.COORD_MAX)
                        next_x, next_y = raw_values[4], raw_values[5]
                        geom_params_normalized[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                elif cmd_char.lower() in ['q', 's', 't']: # Simplified S, T handling
                    if len(raw_values) >= 4:
                        geom_params_normalized[2] = self._normalize(raw_values[0], self.COORD_MIN, self.COORD_MAX) # Control point 1
                        geom_params_normalized[3] = self._normalize(raw_values[1], self.COORD_MIN, self.COORD_MAX)
                        next_x, next_y = raw_values[2], raw_values[3]
                        geom_params_normalized[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                elif cmd_char.lower() == 'a':
                    if len(raw_values) >= 7:
                        geom_params_normalized[0] = self._normalize(raw_values[0], self.RADIUS_MIN, self.RADIUS_MAX) # rx
                        geom_params_normalized[1] = self._normalize(raw_values[1], self.RADIUS_MIN, self.RADIUS_MAX) # ry
                        geom_params_normalized[2] = self._normalize(raw_values[2], self.ROT_MIN, self.ROT_MAX)     # x-axis-rotation
                        geom_params_normalized[3] = self._normalize(raw_values[3], self.FLAG_MIN, self.FLAG_MAX)    # large-arc-flag
                        geom_params_normalized[4] = self._normalize(raw_values[4], self.FLAG_MIN, self.FLAG_MAX)    # sweep-flag
                        next_x, next_y = raw_values[5], raw_values[6]
                        geom_params_normalized[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                        geom_params_normalized[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                elif cmd_char.lower() == 'z':
                    next_x, next_y = subpath_start_x, subpath_start_y
                    geom_params_normalized[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                    geom_params_normalized[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                
                current_x, current_y = next_x, next_y
                row_params[:self.num_coord_params] = geom_params_normalized
                # First two columns are float versions of rho and tau
                current_row = torch.cat((torch.tensor([float(element_rho), float(cmd_tau)]), row_params))
                sequence_rows.append(current_row)

            # Style row for the path
            style_params_normalized = self._get_style_params_normalized(style_data)
            # Ensure style_params_normalized fits into self.num_parameter_slots
            padded_style_params = torch.full((self.num_parameter_slots,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
            padded_style_params[:self.num_style_params] = style_params_normalized
            
            style_row_values = torch.cat((torch.tensor([float(element_rho), float(self.PATH_COMMAND_TYPES['STYLE'])]), padded_style_params))
            sequence_rows.append(style_row_values)

        elif element_type_str in ['rect', 'circle', 'ellipse'] and commands_data:
            # 1. Definition Row
            def_params_geom_norm = torch.full((self.num_coord_params,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
            attrs = commands_data
            if element_type_str == 'rect':
                def_params_geom_norm[0] = self._normalize(attrs.get('x', 0), self.COORD_MIN, self.COORD_MAX)
                def_params_geom_norm[1] = self._normalize(attrs.get('y', 0), self.COORD_MIN, self.COORD_MAX)
                def_params_geom_norm[2] = self._normalize(attrs.get('rx', 0), self.RADIUS_MIN, self.RADIUS_MAX)
                def_params_geom_norm[3] = self._normalize(attrs.get('ry', 0), self.RADIUS_MIN, self.RADIUS_MAX)
                def_params_geom_norm[4] = self._normalize(attrs.get('width', 0), self.RADIUS_MIN, self.RADIUS_MAX)
                def_params_geom_norm[5] = self._normalize(attrs.get('height', 0), self.RADIUS_MIN, self.RADIUS_MAX)
            elif element_type_str == 'circle':
                def_params_geom_norm[0] = self._normalize(attrs.get('cx', 0), self.COORD_MIN, self.COORD_MAX)
                def_params_geom_norm[1] = self._normalize(attrs.get('cy', 0), self.COORD_MIN, self.COORD_MAX)
                def_params_geom_norm[2] = self._normalize(attrs.get('r', 0), self.RADIUS_MIN, self.RADIUS_MAX)
            elif element_type_str == 'ellipse':
                def_params_geom_norm[0] = self._normalize(attrs.get('cx', 0), self.COORD_MIN, self.COORD_MAX)
                def_params_geom_norm[1] = self._normalize(attrs.get('cy', 0), self.COORD_MIN, self.COORD_MAX)
                def_params_geom_norm[2] = self._normalize(attrs.get('rx', 0), self.RADIUS_MIN, self.RADIUS_MAX)
                def_params_geom_norm[3] = self._normalize(attrs.get('ry', 0), self.RADIUS_MIN, self.RADIUS_MAX)

            padded_def_params = torch.full((self.num_parameter_slots,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
            padded_def_params[:self.num_coord_params] = def_params_geom_norm
            def_row_values = torch.cat((torch.tensor([float(element_rho), float(self.PATH_COMMAND_TYPES['DEF'])]), padded_def_params))
            sequence_rows.append(def_row_values)

            # 2. Style Row
            style_params_normalized = self._get_style_params_normalized(style_data)
            padded_style_params = torch.full((self.num_parameter_slots,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
            padded_style_params[:self.num_style_params] = style_params_normalized
            style_row_values = torch.cat((torch.tensor([float(element_rho), float(self.PATH_COMMAND_TYPES['STYLE'])]), padded_style_params))
            sequence_rows.append(style_row_values)

        
        if sequence_rows:
            return torch.stack(sequence_rows) # Output will be [NumRowsForElement, 1+1+num_parameter_slots]
        

        return None


class SVGToTensor_Normalized:
    def __init__(self, target_norm_min=-1.0, target_norm_max=1.0):
        self.num_geom_params = 8  # µ₀, ν₀, µ₁, ν₁, µ₂, ν₂, µ₃, ν₃
        self.num_fill_style_params = 3 # R, G, B, α (for fill)
        
        # Output columns: element_type_idx, cmd_seq_idx_within_element, cmd_type_idx, 
        #                 + 8 geometry params + 4 fill style params
        self.output_matrix_cols = 3 + self.num_geom_params + self.num_fill_style_params # 3 + 8 + 4 = 15

        # Parameter ranges
        self.COORD_MIN, self.COORD_MAX = -128.0, 127.0 # Expanded slightly based on typical SVG canvas sizes
        self.RADIUS_MIN, self.RADIUS_MAX = 0.0, 128.0 # For r, rx, ry, width, height
        self.FLAG_MIN, self.FLAG_MAX = 0.0, 1.0       # Arc flags
        self.ROT_MIN, self.ROT_MAX = -180.0, 180.0    # Arc x-axis-rotation (SVG allows any real, often clamped)
        self.OPACITY_MIN, self.OPACITY_MAX = 0.0, 1.0
        self.COLOR_MIN, self.COLOR_MAX = 0.0, 255.0
        # STROKEW_MIN, STROKEW_MAX are not used as we only encode fill style here

        # Add this line to fix the error:
        self.num_parameter_slots = max(self.num_geom_params, self.num_fill_style_params)

        self.target_norm_min = target_norm_min
        self.target_norm_max = target_norm_max

        num_bins = 256 # Number of bins for quantization
        self.num_bins = num_bins
        self.bin_max_idx = num_bins - 1

        # Element type indices (ρ in the new scheme)
        # Using 0-indexed values. The image's example values (1,6,4,7,2) might be from a different mapping.
        self.ELEMENT_TYPES = {'<BOS>':0, 'rect': 1, 'circle': 2, 'ellipse': 3, 'path': 4, '<EOS>':5, '<PAD>':6 }
        self.PAD_ELEMENT_IDX = self.ELEMENT_TYPES['<PAD>']

        # Command type indices (cmd in the new scheme)
        self.PATH_COMMAND_TYPES = {
            'NO_CMD': 0, # For BOS, EOS, PAD, or if no specific command applies
            'm': 1, 'l': 2, 'h': 3, 'v': 4, 'c': 5, 's': 6, 'q': 7, 't': 8, 'a': 9, 'z': 10,
               'CIRCLE': 11, 'ELLIPSE': 12, 'RECT': 13
        }
        self.NO_CMD_IDX = self.PATH_COMMAND_TYPES['NO_CMD']
        #self.DEF_CMD_IDX = self.PATH_COMMAND_TYPES['DEF']
        
        self.DEFAULT_PARAM_VAL = 0.0 # Default for unused normalized parameter slots
        self.DEFAULT_PARAM_VAL_NORM = 0.001956947162426559
        self.DEFAULT_RGB_OPX_NORM = -1.0

    def _normalize_v1(self, value, val_min, val_max):
        value = float(value)
        val_min, val_max = float(val_min), float(val_max)
        if val_max == val_min: # Avoid division by zero
            # If min and max are same, value should be that fixed point.
            # Normalized, this means it's either target_norm_min or target_norm_max or middle.
            # Let's map it to the middle of the target range.
            if value == val_min: # Or handle based on where value lies if it's outside
                 return (self.target_norm_min + self.target_norm_max) / 2.0 # Or map to a specific point
            # This case implies a constant feature, could return 0 or middle of target range.
            return (self.target_norm_min + self.target_norm_max) / 2.0


        value_clamped = max(val_min, min(value, val_max)) # Clamp to original range
        if val_min == -128.0:
            return value_clamped + abs(val_min) # Adjust for negative min
        else:
            return value_clamped
        #norm_0_1 = (value_clamped - val_min) / (val_max - val_min)
        #return norm_0_1 * (self.target_norm_max - self.target_norm_min) + self.target_norm_min
        #return value

    def _normalize(self, value, min_val, max_val):
        ### actually quanitizing
        
        """Quantizes value to [0, num_bins-1]"""
        #print(value, min_val, max_val)
        if max_val == min_val: return 1
        # Ensure value is tensor for clamp
        value_tensor = torch.tensor(value, dtype=torch.float32)
        value_clamped = torch.clamp(value_tensor, min_val, max_val)
        # Prevent division by zero or negative range
        range_val = max(max_val - min_val, 1e-9)
        value_shifted = value_clamped - min_val
        # Calculate bin index carefully
        bin_index = torch.floor(value_shifted / range_val * self.num_bins)
        # Final clamp to ensure index is within bounds
        bin_index = torch.clamp(bin_index, 0, self.bin_max_idx)
        return bin_index.long().item()

    def _get_fill_style_params_normalized(self, style_data):
        """Normalizes fill color (RGB) and fill opacity (A) parameters."""
        params_fill_style = torch.full((self.num_fill_style_params,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
        
        fill_rgb = style_data.get('fill_rgb', [0,0,0]) 
        
        # Determine fill opacity: use 'fill-opacity' first, then 'opacity'. Default to 1.0.
        # If 'fill' is 'none', opacity is effectively 0 for the fill.
        fill_val_str = str(style_data.get('fill', 'black')).lower() # Default fill is black in SVG

        if fill_val_str == 'none':
            fill_opacity = 0.0
            # fill_rgb can remain as is (e.g. [0,0,0]) or be set to a specific value for "none"
        else:
            # If fill is not 'none', parse opacity.
            # fill-opacity overrides general opacity for fill.
            if 'fill-opacity' in style_data:
                fill_opacity = style_data['fill-opacity']
            elif 'opacity' in style_data: # General opacity applies to fill if fill-opacity is not set
                fill_opacity = style_data['opacity']
            else:
                fill_opacity = 1.0 # Default opaque

        params_fill_style[0] = self._normalize(fill_rgb[0], self.COLOR_MIN, self.COLOR_MAX) # R
        params_fill_style[1] = self._normalize(fill_rgb[1], self.COLOR_MIN, self.COLOR_MAX) # G
        params_fill_style[2] = self._normalize(fill_rgb[2], self.COLOR_MIN, self.COLOR_MAX) # B
        #params_fill_style[3] = self._normalize(fill_opacity, self.OPACITY_MIN, self.OPACITY_MAX) # Alpha (fill-opacity)
        
        return params_fill_style

    def create_tensor_for_element(self, element_data):
        element_type_str = element_data['type']
        # For path, commands_data is list of path cmds. For shapes, it's a dict of attrs.
        commands_data = element_data.get('commands') 
        style_data = element_data.get('style', {})

        element_rho_idx = self.ELEMENT_TYPES.get(element_type_str, self.PAD_ELEMENT_IDX)
        # element_rho_norm = self._normalize(
        #                         element_rho_idx,
        #                         min(self.ELEMENT_TYPES.values()),
        #                         max(self.ELEMENT_TYPES.values())
        #                     )
        if element_rho_idx == self.PAD_ELEMENT_IDX and element_type_str not in ['<BOS>', '<EOS>', '<PAD>']:
            return None # Skip unknown element types

        sequence_rows = []
        fill_style_params_norm = self._get_fill_style_params_normalized(style_data)

        if element_type_str == 'path' and commands_data:
            #print(commands_data)
            current_x, current_y = 0.0, 0.0 
            subpath_start_x, subpath_start_y = 0.0, 0.0

            for cmd_seq_idx, cmd_dict in enumerate(commands_data): # cmd_seq_idx is our 'τ'
                geom_params_norm = torch.full((self.num_geom_params,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
                
                cmd_char = cmd_dict['command']
                cmd_code = self.PATH_COMMAND_TYPES.get(cmd_char.lower(), self.NO_CMD_IDX)
                # cmd_code_norm = self._normalize(
                #                         cmd_code,
                #                         min(self.PATH_COMMAND_TYPES.values()),
                #                         max(self.PATH_COMMAND_TYPES.values())
                #                     )
                raw_values = cmd_dict.get('values', [])
                
                # µ₀, ν₀ = start point of the current segment (current_x, current_y before this command)
                # For 'M' command, this is handled slightly differently as it sets a new current_x, current_y
                # which becomes the start point AND end point for the 'M' operation itself.
                
                temp_current_x, temp_current_y = current_x, current_y # Save state before modification by relative commands

                next_x, next_y = current_x, current_y 

                if cmd_char.lower() == 'm':
                    if len(raw_values) >= 2:
                        dx, dy = raw_values[0], raw_values[1]
                        next_x = dx
                        next_y = dy
                        # For M, µ₀,ν₀ and µ₃,ν₃ are the new point.
                        geom_params_norm[0] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX) 
                        geom_params_norm[1] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX) 
                        geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX) 
                        geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX) 
                        subpath_start_x, subpath_start_y = next_x, next_y
                else: # For L, C, Q, A, Z, H, V: µ₀, ν₀ is current_x, current_y
                    geom_params_norm[0] = self._normalize(current_x, self.COORD_MIN, self.COORD_MAX)
                    geom_params_norm[1] = self._normalize(current_y, self.COORD_MIN, self.COORD_MAX)

                    if cmd_char.lower() == 'l':
                        if len(raw_values) >= 2:
                            dx, dy = raw_values[0], raw_values[1]
                            next_x = dx
                            next_y = dy
                            geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                    elif cmd_char.lower() == 'h':
                        if len(raw_values) >= 1:
                            dx = raw_values[0]
                            next_x = dx
                            # next_y remains temp_current_y
                            geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[7] = self._normalize(temp_current_y, self.COORD_MIN, self.COORD_MAX)
                    elif cmd_char.lower() == 'v':
                        if len(raw_values) >= 1:
                            dy = raw_values[0]
                            next_y = dy
                            # next_x remains temp_current_x
                            geom_params_norm[6] = self._normalize(temp_current_x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                    elif cmd_char.lower() == 'c':
                        if len(raw_values) >= 6:
                            c1x, c1y, c2x, c2y, ex, ey = raw_values
                            # if cmd_char.islower(): 
                            #     c1x+=temp_current_x; c1y+=temp_current_y; c2x+=temp_current_x; c2y+=temp_current_y; ex+=temp_current_x; ey+=temp_current_y
                            geom_params_norm[2] = self._normalize(c1x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[3] = self._normalize(c1y, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[4] = self._normalize(c2x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[5] = self._normalize(c2y, self.COORD_MIN, self.COORD_MAX)
                            next_x, next_y = ex, ey
                            geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                    elif cmd_char.lower() in ['q', 's']: # Assuming S gets simplified to Q-like params by parser or here
                        if len(raw_values) >= 4: # (c1x, c1y, ex, ey)
                            c1x, c1y, ex, ey = raw_values
                            # if cmd_char.islower():
                            #     c1x+=temp_current_x; c1y+=temp_current_y; ex+=temp_current_x; ey+=temp_current_y
                            geom_params_norm[2] = self._normalize(c1x, self.COORD_MIN, self.COORD_MAX) 
                            geom_params_norm[3] = self._normalize(c1y, self.COORD_MIN, self.COORD_MAX) 
                            # µ₂, ν₂ (geom_params_norm[4,5]) remain default for Q
                            next_x, next_y = ex, ey
                            geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                    elif cmd_char.lower() in ['t']: # Assuming T gets simplified to Q-like params by parser or here
                         if len(raw_values) >= 2: # (ex, ey) - control point is implicit
                            ex, ey = raw_values
                            # if cmd_char.islower():
                            #     ex+=temp_current_x; ey+=temp_current_y
                            # For T, control point is reflection of previous Q's control point.
                            # This simplified version doesn't calculate it, sets cp1 to current point.
                            geom_params_norm[2] = self._normalize(temp_current_x, self.COORD_MIN, self.COORD_MAX) 
                            geom_params_norm[3] = self._normalize(temp_current_y, self.COORD_MIN, self.COORD_MAX)
                            next_x, next_y = ex, ey
                            geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                            geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                    elif cmd_char.lower() == 'a':
                        if len(raw_values) >= 7:
                            svg_rx, svg_ry, svg_x_axis_rot, svg_large_arc_flag, svg_sweep_flag, svg_ex, svg_ey = raw_values
                            next_x, next_y = svg_ex, svg_ey
                            # if cmd_char.islower(): 
                            #     next_x += temp_current_x; next_y += temp_current_y
                            #geom_params_norm[0] = temp_current_y
                            geom_params_norm[0] = self._normalize(svg_rx, self.RADIUS_MIN, self.RADIUS_MAX)           # µ₁ = rx
                            geom_params_norm[1] = self._normalize(svg_ry, self.RADIUS_MIN, self.RADIUS_MAX)           # ν₁ = ry
                            geom_params_norm[2] = self._normalize(svg_x_axis_rot, self.ROT_MIN, self.ROT_MAX)         # µ₂ = x-axis-rotation
                            geom_params_norm[3] = self._normalize(svg_large_arc_flag, self.FLAG_MIN, self.FLAG_MAX)  # ν₂ = large-arc-flag
                            geom_params_norm[4] = self._normalize(svg_sweep_flag, self.FLAG_MIN, self.FLAG_MAX)  # ν₂ = sweep-flag
                            geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)           # µ₃ = end_x
                            geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)           # ν₃ = end_y
                    elif cmd_char.lower() == 'z':
                        next_x, next_y = subpath_start_x, subpath_start_y
                        geom_params_norm[6] = self._normalize(next_x, self.COORD_MIN, self.COORD_MAX)
                        geom_params_norm[7] = self._normalize(next_y, self.COORD_MIN, self.COORD_MAX)
                
                current_x, current_y = next_x, next_y
                
                cmd_seq_idx = torch.tensor([1], dtype=torch.float32)
                indices = torch.tensor([element_rho_idx,  cmd_code], dtype=torch.float32)
                current_row = torch.cat((indices, geom_params_norm, fill_style_params_norm, cmd_seq_idx))
                sequence_rows.append(current_row)

        elif element_type_str in ['rect', 'circle', 'ellipse']:
            #print(commands_data)
            cmd_seq_idx = 1 # 'τ' = 0 for shapes (single definition row)
            cmd_code = 0
            
            geom_params_norm = torch.full((self.num_geom_params,), self.DEFAULT_PARAM_VAL, dtype=torch.float32)
            #attrs = commands_data # This is a dict of attributes
            attrs = {item['command']: item['values'] for item in commands_data}

            if element_type_str == 'rect':
                cmd_code = self.PATH_COMMAND_TYPES['RECT']
                # cmd_code_norm = self._normalize(
                #                         cmd_code,
                #                         min(self.PATH_COMMAND_TYPES.values()),
                #                         max(self.PATH_COMMAND_TYPES.values())
                #                     )
                # µ₀,ν₀ = x,y; µ₁,ν₁ = rx,ry; µ₂,ν₂ = width,height
                geom_params_norm[0] = self._normalize(attrs.get('x', 0.0), self.COORD_MIN, self.COORD_MAX)
                geom_params_norm[1] = self._normalize(attrs.get('y', 0.0), self.COORD_MIN, self.COORD_MAX)
                geom_params_norm[2] = self._normalize(attrs.get('rx', 0.0), self.RADIUS_MIN, self.RADIUS_MAX) 
                geom_params_norm[3] = self._normalize(attrs.get('ry', 0.0), self.RADIUS_MIN, self.RADIUS_MAX) 
                geom_params_norm[4] = self._normalize(attrs.get('width', 0.0), self.RADIUS_MIN, self.RADIUS_MAX)
                geom_params_norm[5] = self._normalize(attrs.get('height', 0.0), self.RADIUS_MIN, self.RADIUS_MAX)
            elif element_type_str == 'circle':
                cmd_code = self.PATH_COMMAND_TYPES['CIRCLE']
                # cmd_code_norm = self._normalize(
                #                         cmd_code,
                #                         min(self.PATH_COMMAND_TYPES.values()),
                #                         max(self.PATH_COMMAND_TYPES.values())
                #                     )
                # µ₀,ν₀ = cx,cy; µ₁ = r
                geom_params_norm[0] = self._normalize(attrs.get('cx', 0.0), self.COORD_MIN, self.COORD_MAX)
                geom_params_norm[1] = self._normalize(attrs.get('cy', 0.0), self.COORD_MIN, self.COORD_MAX)
                geom_params_norm[2] = self._normalize(attrs.get('r', 0.0), self.RADIUS_MIN, self.RADIUS_MAX) 
            elif element_type_str == 'ellipse':
                cmd_code = self.PATH_COMMAND_TYPES['ELLIPSE']
                # cmd_code_norm = self._normalize(
                #                         cmd_code,
                #                         min(self.PATH_COMMAND_TYPES.values()),
                #                         max(self.PATH_COMMAND_TYPES.values())
                #                     )
                # µ₀,ν₀ = cx,cy; µ₁,ν₁ = rx,ry
                geom_params_norm[0] = self._normalize(attrs.get('cx', 0.0), self.COORD_MIN, self.COORD_MAX)
                geom_params_norm[1] = self._normalize(attrs.get('cy', 0.0), self.COORD_MIN, self.COORD_MAX)
                geom_params_norm[2] = self._normalize(attrs.get('rx', 0.0), self.RADIUS_MIN, self.RADIUS_MAX)
                geom_params_norm[3] = self._normalize(attrs.get('ry', 0.0), self.RADIUS_MIN, self.RADIUS_MAX)
            
            cmd_seq_idx = torch.tensor([1], dtype=torch.float32)
            indices = torch.tensor([element_rho_idx,  cmd_code], dtype=torch.float32)
            current_row = torch.cat((indices, geom_params_norm, fill_style_params_norm, cmd_seq_idx))
            sequence_rows.append(current_row)
        
        if sequence_rows:
            return torch.stack(sequence_rows)
        return None


def convert_svg_to_tensor(svg_file_path: str, max_seq_length=256):
    tensor_converter = SVGToTensor_Normalized()
    output_matrix_cols = tensor_converter.output_matrix_cols 
    
    PAD_GEOM_PARAMS = torch.full((tensor_converter.num_geom_params,), tensor_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)
    PAD_FILL_STYLE_PARAMS = torch.full((tensor_converter.num_fill_style_params,), tensor_converter.DEFAULT_PARAM_VAL, dtype=torch.float32)
    
    PAD_INDICES = torch.tensor([
        float(tensor_converter.PAD_ELEMENT_IDX), 
        0.0, # cmd_seq_idx for PAD
        float(tensor_converter.NO_CMD_IDX) # cmd_type_idx for PAD
    ], dtype=torch.float32)
    PADDING_ROW_TEMPLATE = torch.cat((PAD_INDICES, PAD_GEOM_PARAMS, PAD_FILL_STYLE_PARAMS))

    parser = SVGParser()
    try:
        # Ensure SVGParser().parse_svg() can take a file path string
        parsed_data = parser.parse_svg(svg_file_path) 
        if not parsed_data or not parsed_data.get('elements'):
            # print(f"Warning: No elements found in {svg_file_path}. Returning padding tensor.")
            return PADDING_ROW_TEMPLATE.unsqueeze(0).repeat(max_seq_length, 1)
    except Exception as e:
        print(f"FATAL Error parsing {svg_file_path}: {e}. Returning padding tensor.")
        return PADDING_ROW_TEMPLATE.unsqueeze(0).repeat(max_seq_length, 1)

    element_tensors = []
    for element_dict in parsed_data.get('elements', []):
        tensor_rows = tensor_converter.create_tensor_for_element(element_dict)
        if tensor_rows is not None and tensor_rows.shape[0] > 0:
            element_tensors.append(tensor_rows)

    if not element_tensors:
        # print(f"Warning: No convertible elements in {svg_file_path}. Returning padding tensor.")
        return PADDING_ROW_TEMPLATE.unsqueeze(0).repeat(max_seq_length, 1)

    try:
        combined_tensor = torch.cat(element_tensors, dim=0)
    except RuntimeError as e: # Catches issues if element_tensors is empty or contains non-tensors
        print(f"Error concatenating element tensors for {svg_file_path}: {e}. Returning padding tensor.")
        return PADDING_ROW_TEMPLATE.unsqueeze(0).repeat(max_seq_length, 1)


    SOS_INDICES = torch.tensor([
        float(tensor_converter.ELEMENT_TYPES['<BOS>']), 
        0.0, 
        float(tensor_converter.NO_CMD_IDX)
    ], dtype=torch.float32)
    SOS_TOKEN = torch.cat((SOS_INDICES, PAD_GEOM_PARAMS, PAD_FILL_STYLE_PARAMS)) # BOS uses default params

    EOS_INDICES = torch.tensor([
        float(tensor_converter.ELEMENT_TYPES['<EOS>']),
        0.0, 
        float(tensor_converter.NO_CMD_IDX)
    ], dtype=torch.float32)
    EOS_TOKEN = torch.cat((EOS_INDICES, PAD_GEOM_PARAMS, PAD_FILL_STYLE_PARAMS)) # EOS uses default params

    final_sequence = torch.cat([SOS_TOKEN.unsqueeze(0), combined_tensor, EOS_TOKEN.unsqueeze(0)], dim=0)
    seq_len = final_sequence.shape[0]

    if seq_len > max_seq_length:
        tensor_output = final_sequence[:max_seq_length]
        # Ensure last token is EOS if truncated, and it wasn't already EOS
        if tensor_output[-1, 0] != float(tensor_converter.ELEMENT_TYPES['<EOS>']):
            tensor_output[-1] = EOS_TOKEN 
    elif seq_len < max_seq_length:
        padding_rows_count = max_seq_length - seq_len
        padding_tensor = PADDING_ROW_TEMPLATE.unsqueeze(0).repeat(padding_rows_count, 1)
        tensor_output = torch.cat([final_sequence, padding_tensor], dim=0)
    else: # seq_len == max_seq_length
        tensor_output = final_sequence
    
    return tensor_output

# --- convert_svg_to_tensor function ---
def convert_svg_to_tensor(svg_file, max_seq_length=256):
    # num_bins is not needed if we normalize to continuous values
    tensor_converter = SVGToTensor_Normalized()
    output_matrix_cols = tensor_converter.output_matrix_cols
    PADDING_ELEMENT_IDX = float(tensor_converter.PAD_ELEMENT_IDX)
    NO_CMD_IDX_FLOAT = float(tensor_converter.NO_CMD_IDX)
    DEFAULT_PARAM_VAL_FLOAT = tensor_converter.DEFAULT_PARAM_VAL

    parser = SVGParser()
    try:
        parsed_data = parser.parse_svg(svg_file)
        if not parsed_data or not parsed_data.get('elements'):
            return torch.full((max_seq_length, output_matrix_cols), DEFAULT_PARAM_VAL_FLOAT, dtype=torch.float32)
    except Exception as e:
        print(f"FATAL Error parsing {svg_file}: {e}. Returning default tensor.")
        return torch.full((max_seq_length, output_matrix_cols), DEFAULT_PARAM_VAL_FLOAT, dtype=torch.float32)

    element_tensors = []
    for element_dict in parsed_data.get('elements', []):
        tensor_rows = tensor_converter.create_tensor_for_element(element_dict)
        if tensor_rows is not None and tensor_rows.shape[0] > 0:
            element_tensors.append(tensor_rows)

    if not element_tensors:
        return torch.full((max_seq_length, output_matrix_cols), DEFAULT_PARAM_VAL_FLOAT, dtype=torch.float32)

    try:
        combined_tensor = torch.cat(element_tensors, dim=0)
    except Exception as e:
        print(f"Error concatenating element tensors for {svg_file}: {e}. Returning default tensor.")
        return torch.full((max_seq_length, output_matrix_cols), DEFAULT_PARAM_VAL_FLOAT, dtype=torch.float32)

    # SOS/EOS tokens
    sos_eos_params = torch.full((tensor_converter.num_parameter_slots,), DEFAULT_PARAM_VAL_FLOAT, dtype=torch.float32)
    sos_token = torch.cat((torch.tensor([float(tensor_converter.ELEMENT_TYPES['<BOS>']), NO_CMD_IDX_FLOAT]), sos_eos_params))
    eos_token = torch.cat((torch.tensor([float(tensor_converter.ELEMENT_TYPES['<EOS>']), NO_CMD_IDX_FLOAT]), sos_eos_params))

    final_sequence = torch.cat([sos_token.unsqueeze(0), combined_tensor, eos_token.unsqueeze(0)], dim=0)
    seq_len = final_sequence.shape[0]

    # Pad or truncate
    if seq_len == 0:
        tensor_output = torch.full((max_seq_length, output_matrix_cols), PADDING_ELEMENT_IDX, dtype=torch.float32)
        tensor_output[:,1:] = DEFAULT_PARAM_VAL_FLOAT # Ensure params are default float
    elif seq_len > max_seq_length:
        tensor_output = final_sequence[:max_seq_length]
        if tensor_output[-1, 0] != float(tensor_converter.ELEMENT_TYPES['<EOS>']): # Ensure last token is EOS if truncated
            tensor_output[-1,0] = float(tensor_converter.ELEMENT_TYPES['<EOS>'])
            tensor_output[-1,1] = NO_CMD_IDX_FLOAT
            tensor_output[-1,2:] = DEFAULT_PARAM_VAL_FLOAT
    elif seq_len < max_seq_length:
        padding_rows = max_seq_length - seq_len
        padding_row_template_params = torch.full((tensor_converter.num_parameter_slots,), DEFAULT_PARAM_VAL_FLOAT, dtype=torch.float32)
        padding_row_template = torch.cat((torch.tensor([PADDING_ELEMENT_IDX, NO_CMD_IDX_FLOAT]), padding_row_template_params))
        padding = padding_row_template.unsqueeze(0).repeat(padding_rows, 1)
        tensor_output = torch.cat([final_sequence, padding], dim=0)
    else: # seq_len == max_seq_length
        tensor_output = final_sequence
    
    # The output tensor now contains floats:
    # Col 0: float(element_rho_idx)
    # Col 1: float(command_tau_idx)
    # Col 2 onwards: normalized continuous parameters in [-1, 1]
    return tensor_output