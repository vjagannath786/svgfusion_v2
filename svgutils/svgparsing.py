import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Tuple


class SVGConstraints:
    # Valid SVG elements
    VALID_ELEMENTS = {
        'svg',
        'g',
        'path',
        'clipPath',
        'rect',
        'circle',
        'ellipse',
        #'line',
        #'polyline',
        #'polygon',
        'title',
    }

    max_svg_size: int = 10000

    # Valid attributes for each element
    VALID_ATTRIBUTES = {
        'svg': {
            'width', 'height', 'viewBox', 'xmlns', 'xmlns:xlink',
            'version', 'preserveAspectRatio', 'class'
        },
        'g': {
            'transform', 'fill', 'stroke', 'clip-path', 'opacity',
            'stroke-width', 'style'
        },
        'path': {
            'd', 'fill', 'stroke', 'stroke-width', 'opacity',
            'transform', 'style', 'stroke-linecap', 'stroke-linejoin', 'id',
            'fill-rule', 'clip-rule', 'stroke-miterlimit','fill-opacity'
        },
        'clipPath': {
            'id', 'transform'
        },
        'circle' : {
            'cx', 'cy', 'r', 'fill', 'stroke-linecap', 'stroke', 'transform','stroke-width','style', 'stroke-linejoin', 'stroke-miterlimit'
        },
        'ellipse' : {
            'cx', 'cy', 'rx', 'ry','fill', 'stroke-linecap', 'stroke', 'transform','stroke-width','style', 'stroke-linejoin', 'stroke-miterlimit', 'opacity'
        },
        'rect' : {
            'x', 'y', 'width', 'height', 'rx', 'ry', 'fill', 'style', 'stroke-linecap', 'stroke-linejoin', 'id',
            'fill-rule', 'clip-rule', 'transform', 'stroke-width', 'stroke','stroke-miterlimit','fill-opacity','opacity'
        }
    }

    # Valid path commands
    VALID_PATH_COMMANDS = {
        'M', 'm',  # moveto
        'L', 'l',  # lineto
        'H', 'h',  # horizontal lineto
        'V', 'v',  # vertical lineto
        'C', 'c',  # curveto
        'S', 's',  # smooth curveto
        'Q', 'q',  # quadratic Bézier curve
        'T', 't',  # smooth quadratic Bézier curve
        'A', 'a',  # elliptical Arc
        'Z', 'z'   # closepath
    }

    # Valid transform functions
    VALID_TRANSFORMS = {
        'translate',
        'scale',
        'rotate',
        'skewX',
        'skewY',
        'matrix'
    }

    @classmethod
    def validate_element(cls, element_name):
        return element_name.split('}')[-1] in cls.VALID_ELEMENTS

    @classmethod
    def validate_attributes(cls, element_name, attributes):
        element_name = element_name.split('}')[-1]
        if element_name not in cls.VALID_ATTRIBUTES:
            return False, list(attributes)  # If element is not in valid attributes, all are invalid.
    
        invalid_attrs = [attr for attr in attributes if attr not in cls.VALID_ATTRIBUTES[element_name]]
        return len(invalid_attrs) == 0, invalid_attrs

    @classmethod
    def validate_path_commands(cls, commands):
        return all(cmd in cls.VALID_PATH_COMMANDS for cmd in commands)



def extract_numeric_value(value_str):
    """Extract numeric value from string with units (px, %, etc.)"""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    # Remove all units (px, pt, em, etc.) and whitespace
    value_str = str(value_str).strip().lower()
    numeric_str = ''.join(c for c in value_str if c.isdigit() or c in '.-')
    
    try:
        return int(numeric_str)
    except ValueError:
        return None

class SVGParser:
    def __init__(self):
        self.svg_ns = "{http://www.w3.org/2000/svg}"
        
        # Command definitions with expected number of values
        self.path_commands = {
            'M': 2, 'm': 2,  # moveto
            'L': 2, 'l': 2,  # lineto
            'H': 1, 'h': 1,  # horizontal lineto
            'V': 1, 'v': 1,  # vertical lineto
            'C': 6, 'c': 6,  # curveto
            'S': 4, 's': 4,  # smooth curveto
            'Q': 4, 'q': 4,  # quadratic Bézier
            'T': 2, 't': 2,  # smooth quadratic Bézier
            'A': 7, 'a': 7,  # elliptical arc
            'Z': 0, 'z': 0   # closepath
        }

        self.constraints = SVGConstraints()

    def parse_svg(self, svg_file: str) -> Dict:
        """Parse SVG file and extract element commands and values."""
        tree = ET.parse(svg_file)
        root = tree.getroot()


        ### Get height, width and viewBox details
        viewport_info = {
            'width': root.get('width', '100%'),
            'height': root.get('height', '100%'),
            'viewBox': root.get('viewBox', '0 0 100 100')
        }

        # Convert viewBox string to numbers
        try:
            viewBox = viewport_info['viewBox'].split()
            viewport_info['viewBox'] = {
                'min_x': viewBox[0],
                'min_y': viewBox[1],
                'width': viewBox[2],
                'height': viewBox[3]
            }
        except (IndexError, ValueError):
            viewport_info['viewBox'] = {
                'min_x': 0,
                'min_y': 0,
                'width': 128,
                'height': 128
            }

        
        # Try to convert width and height to numbers if they're not percentages
        for dim in ['width', 'height']:
            value = viewport_info[dim]
            if isinstance(value, str):
                if value.endswith('%'):
                    # Handle percentage
                    viewport_info[dim] = viewport_info['viewBox'][dim]
                else:
                    # Handle px, pt, em, or any other unit
                    numeric_value = extract_numeric_value(value)
                    if numeric_value is not None:
                        viewport_info[dim] = numeric_value
                    else:
                        viewport_info[dim] = viewport_info['viewBox'][dim]




        #print(viewport_info)
        
        # parsed_data = {
        #     'viewport': viewport_info,
        #     'paths': [],
        #     'circles': [],
        #     'rects': [],
        #     'ellipses': []
        # }

        parsed_data = {
            'viewport': viewport_info,
            'elements': []
        }

        

        
        if len(svg_file) > self.constraints.max_svg_size:
            print("SVG file exceeded max length")
            return []


        
        for element in root:
            tag = element.tag.replace(self.svg_ns, '')

            #print(element)

            # Validate element before processing
            if not self.constraints.validate_element(tag):
                if tag == 'style':
                    raise ValueError(f"Invalid SVG Element '{tag}' found. Raising Error")
                else:
                    print(f"Warning: Invalid SVG element '{tag}' found. Skipping.")
                    continue


            
            # if tag == 'path':
            #     path_data = self.parse_path_element(element)
            #     parsed_data['paths'].append(path_data)
            # elif tag == 'circle':
            #     circle_data = self.parse_circle_element(element)
            #     parsed_data['circles'].append(circle_data)
            # elif tag == 'rect':
            #     rect_data = self.parse_rect_element(element)
            #     parsed_data['rects'].append(rect_data)
            # elif tag == 'ellipse':
            #     rect_data = self.parse_ellipse_element(element)
            #     parsed_data['ellipses'].append(rect_data)

            element_data = {'type': tag}  # Keep track of the type for ordered processing

            if tag == 'path':
                element_data.update(self.parse_path_element(element))
            elif tag == 'circle':
                element_data.update(self.parse_circle_element(element))
            elif tag == 'rect':
                element_data.update(self.parse_rect_element(element))
            elif tag == 'ellipse':
                element_data.update(self.parse_ellipse_element(element))

            parsed_data['elements'].append(element_data)  # Preserve order
            
                
        return parsed_data

    def parse_path_element(self, element) -> Dict:
        """Parse path element and its commands."""

        attributes = element.attrib

        # Validate attributes
        is_valid, invalid_attrs = self.constraints.validate_attributes('path', attributes)
        
        if not is_valid:
            print(f"Warning: Invalid attributes {invalid_attrs} in <path>. Skipping.")
            return {}


        path_data = {
            'commands': [],
            'style': self.parse_style_attributes(element)
        }
        
        d = element.get('d', '')
        if d:
            commands = self.parse_path_commands(d)
            # Validate parsed commands
            if not self.constraints.validate_path_commands([cmd['command'] for cmd in commands]):
                print(f"Warning: Invalid path commands found in <path>. Skipping.")
                return {}
            
            path_data['commands'] = commands
            
        return path_data

    def parse_path_commands(self, path_data: str) -> List[Dict]:
        """Parse path commands and their values."""
        command_pattern = r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)'
        number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        
        commands = []
        matches = re.findall(command_pattern, path_data)
        
        for cmd, values_str in matches:
            values = [float(v) for v in re.findall(number_pattern, values_str)]
            expected_values = self.path_commands[cmd]
            
            # Handle multiple sets of values for the same command
            if expected_values > 0:
                for i in range(0, len(values), expected_values):
                    command_values = values[i:i + expected_values]
                    if len(command_values) == expected_values:
                        commands.append({
                            'command': cmd,
                            'values': command_values,
                            'original': f"{cmd}{','.join(map(str, command_values))}"
                        })
            else:  # Handle Z command
                commands.append({
                    'command': cmd,
                    'values': [],
                    'original': cmd
                })
                
        return commands

    def parse_circle_element(self, element) -> Dict:
        """Parse circle element attributes similar to path commands."""
        attributes = element.attrib
        # Validate attributes
        is_valid, invalid_attrs = self.constraints.validate_attributes('circle', attributes)
        if not is_valid:
            print(f"Warning: Invalid attributes {invalid_attrs} in <circle>. Skipping.")
            return {}



        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        r = float(element.get('r', 0))
        
        commands = [
            {'command': 'cx', 'values': cx, 'original': cx},
            {'command': 'cy', 'values': cy, 'original': cy},
            {'command': 'r', 'values': r, 'original': r},

        ]
        
        return {
            'commands': commands,
            'style': self.parse_style_attributes(element)
        }
    
    def parse_ellipse_element(self, element) -> Dict:
        """Parse circle element attributes similar to path commands."""
        attributes = element.attrib
        # Validate attributes
        is_valid, invalid_attrs = self.constraints.validate_attributes('ellipse', attributes)
        if not is_valid:
            print(f"Warning: Invalid attributes {invalid_attrs} in <ellipse>. Skipping.")
            return {}



        cx = float(element.get('cx', 0))
        cy = float(element.get('cy', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))
        
        commands = [
            {'command': 'cx', 'values': cx, 'original': cx},
            {'command': 'cy', 'values': cy, 'original': cy},
            {'command': 'rx', 'values': rx, 'original': rx},
            {'command': 'ry', 'values': ry, 'original': ry},


        ]
        
        return {
            'commands': commands,
            'style': self.parse_style_attributes(element)
        }

    def parse_rect_element(self, element) -> Dict:
        """Parse rectangle element attributes."""

        attributes = element.attrib

        # Validate attributes
        is_valid, invalid_attrs = self.constraints.validate_attributes('rect', attributes)
        if not is_valid:
            print(f"Warning: Invalid attributes {invalid_attrs} in <rect>. Skipping.")
            return {}




        x = float(element.get('x', 0))
        y = float(element.get('y', 0))
        width = float(element.get('width', 0))
        height = float(element.get('height', 0))
        rx = float(element.get('rx', 0))
        ry = float(element.get('ry', 0))


        commands = [
                {'command': 'x', 'values': x, 'original': x},
                 {'command': 'y', 'values': y, 'original': y},
                  {'command': 'width', 'values': width, 'original': width},
                   {'command': 'height', 'values': height, 'original': height},
                    {'command': 'rx', 'values': rx, 'original': rx},
                    {'command': 'ry', 'values': ry, 'original': ry},

            ]
        



        return {
            'commands': commands,
            'style': self.parse_style_attributes(element)
        }

    def parse_style_attributes(self, element) -> Dict:
        """Parse style attributes of an element."""
        #print(element.get('fill'))
        #print(element.get('fill', 'none'))
        #print(element.get('stroke', 0))
        style = {
            'fill': element.get('fill', 'none'),
            'stroke': element.get('stroke', 'none'),
            'stroke-width': float(element.get('stroke-width', 0)),
            'opacity': element.get('opacity', 0)
        }

        #print(style)
        
        # Convert fill and stroke colors to RGB
        style['fill_rgb'] = self.hex_to_rgb(style['fill'])
        style['stroke_rgb'] = self.hex_to_rgb(style['stroke'])

        #print(style)
        
        return style

    def hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB values."""
        try:

            hex_color = hex_color.lstrip('#')
            if not hex_color or hex_color == 'none':
                return (0, 0, 0)
        
        
            if len(hex_color) == 3:
                hex_color = ''.join([c * 2 for c in hex_color])
            return tuple(int(hex_color[i:i+2], 16)  for i in (0, 2, 4))
        except ValueError:
            print(f"Warning: Invalid hex color: {hex_color}")
            return (0, 0, 0)

# Example usage:
def main():
    parser = SVGParser()
    svg_file = "reconstructed.svg"
    parsed_data = parser.parse_svg(svg_file)
    
    # Print parsed data
    print("\nPaths:")
    for i, path in enumerate(parsed_data['paths']):
        print(f"\nPath {i+1}:")
        for cmd in path['commands']:
            print(f"Command: {cmd['command']}, Values: {cmd['values']}")
        print(f"Style: {path['style']}")
    
    print("\nCircles:")
    for i, circle in enumerate(parsed_data['circles']):
        print(f"\nCircle {i+1}:")
        print(f"Center: ({circle['cx']}, {circle['cy']})")
        print(f"Radius: {circle['r']}")
        print(f"Style: {circle['style']}")

    
    print("\Ellipses:")
    for i, circle in enumerate(parsed_data['ellipses']):
        print(f"\nCircle {i+1}:")
        print(f"Center: ({circle['cx']}, {circle['cy']})")
        print(f"X Radius: {circle['rx']}")
        print(f"Y Radius: {circle['ry']}")
        print(f"Style: {circle['style']}")

    
    
    print("\nRectangles:")
    for i, rect in enumerate(parsed_data['rects']):
        print(f"\nRectangle {i+1}:")
        print(f"Position: ({rect['x']}, {rect['y']})")
        print(f"Size: {rect['width']} x {rect['height']}")
        print(f"Rounded corners: rx={rect['rx']}, ry={rect['ry']}")
        print(f"Style: {rect['style']}")

if __name__ == "__main__":
    main()