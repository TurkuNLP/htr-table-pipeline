import os
import xml.etree.ElementTree as ET
import json

# Define the folders containing the images, XML files, and output
images_folder = 'sample_all_printed'
xml_folder = 'sample-all-printed-xml'
output_folder = 'output'

# Define the namespace for XML parsing (adjust if necessary)
namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# List to hold file pairs
file_list = []

# Function to extract coordinates, baseline, and text from an XML file
def extract_coords_baseline_and_text(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data_list = []
    for cell in root.findall('.//ns:TableCell', namespace):
        coords_elem = cell.find('.//ns:Coords', namespace)
        
        text_line_elem = cell.find('.//ns:TextLine', namespace)
        if text_line_elem is not None:
            text_elem = text_line_elem.find('.//ns:Unicode', namespace)
            if text_elem is not None and text_elem.text:
                text = text_elem.text.strip()
            else:
                text = ''
            baseline_elem = text_line_elem.find('.//ns:Baseline', namespace)
            if baseline_elem is not None:
                baseline = baseline_elem.get('points')
            else:
                baseline = None
        else:
            text = ''
            baseline = None
        
        if coords_elem is not None:
            coords = coords_elem.get('points')
        else:
            coords = ''
        
        data_list.append({
            'coords': coords,
            'baseline': baseline,
            'text': text
        })
    
    return data_list

# Process each image file in the images folder
for image_filename in os.listdir(images_folder):
    if image_filename.endswith('.jpg'):
        base_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(images_folder, image_filename)
        xml_path = os.path.join(xml_folder, base_name + '.xml')
        json_path = os.path.join(output_folder, base_name + '_coords.json')
        
        if os.path.exists(xml_path):
            # Extract coordinates, baseline, and text from the XML file
            data_list = extract_coords_baseline_and_text(xml_path)
            
            # Write coordinates, baseline, and text to a JSON file
            with open(json_path, 'w') as json_file:
                json.dump(data_list, json_file, indent=4)
            
            # Append file pair to list with relative paths
            file_list.append({
                'image': os.path.join(images_folder, image_filename),
                'json': os.path.join(output_folder, base_name + '_coords.json')
            })

            print(f'Processed {base_name} - JSON coordinates, baseline, and text saved to {json_path}')
        else:
            print(f'Warning: XML file not found for {image_filename}')

# Write file list to JSON file
file_list_path = os.path.join(output_folder, 'file_list.json')
with open(file_list_path, 'w') as file_list_file:
    json.dump(file_list, file_list_file, indent=4)

print(f'File list saved to {file_list_path}')