import json
import os
import glob

# Define the class mapping exactly as it appears in dataset.yaml
CLASS_MAPPING = {
    "red_buoy": 0,
    "yellow_buoy": 1,
    "green_buoy": 2
}

def convert_labelme_to_yolo(json_dir, output_dir):
    """
    Converts Labelme JSON annotations to YOLO txt format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            
        img_width = data.get("imageWidth")
        img_height = data.get("imageHeight")
        
        # Output text file will have the same name as the json file
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_path, "w") as out_file:
            for shape in data.get("shapes", []):
                label = shape.get("label")
                
                # Skip shapes not defined in our dataset.yaml
                if label not in CLASS_MAPPING:
                    print(f"Warning: Label '{label}' not in class mapping. Skipping.")
                    continue
                    
                # Skip non-rectangle shapes (like polygons) if any exist
                if shape.get("shape_type") != "rectangle":
                    print(f"Warning: Found non-rectangle shape type '{shape.get('shape_type')}'. Skipping.")
                    continue
                    
                class_id = CLASS_MAPPING[label]
                
                # Labelme stores rectangle points as [[x1, y1], [x2, y2]]
                points = shape.get("points")
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Ensure min/max to avoid negative widths/heights if drawn backwards
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                
                # Calculate YOLO normalized coordinates
                # Center X, Center Y, Width, Height (all between 0 and 1)
                x_center = ((xmin + xmax) / 2.0) / img_width
                y_center = ((ymin + ymax) / 2.0) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Write to the text file
                # Format: <class_id> <x_center> <y_center> <width> <height>
                out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
    print(f"Converted {len(json_files)} files. Output saved to: {output_dir}")

if __name__ == "__main__":
    # SET YOUR DIRECTORIES HERE
    # Directory containing your Labelme .json files
    INPUT_JSON_DIR = "./dataset/labels/val" 
    
    # Directory where you want the YOLO .txt files to be saved
    OUTPUT_YOLO_DIR = "./dataset/labels/val" 
    
    convert_labelme_to_yolo(INPUT_JSON_DIR, OUTPUT_YOLO_DIR)
