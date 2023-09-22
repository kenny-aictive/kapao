import copy
import json
def move_keypoint(input_file_path, output_file_path, keypoint_name, direction, amount):
    """
    Move a specific keypoint in the COCO keypoints annotation file.
    
    Parameters:
        input_file_path (str): Path to the original COCO keypoints annotation file.
        output_file_path (str): Path to save the modified COCO keypoints annotation file.
        keypoint_name (str): Name of the keypoint to move (e.g., "left_shoulder").
        direction (str): Direction to move the keypoint ('x' or 'y').
        amount (int): Amount to move the keypoint.
    """
    # Load the original COCO keypoints annotation file
    with open(input_file_path, 'r') as f:
        original_data = json.load(f)
    
    # Create a deep copy of the original data
    modified_data = copy.deepcopy(original_data)
    
    # Identify the index of the keypoint to move
    keypoint_index = original_data['categories'][0]['keypoints'].index(keypoint_name)
    
    # Iterate through all the annotations to move the specified keypoint
    for annotation in modified_data['annotations']:
        keypoints = annotation['keypoints']
        if direction == 'x':
            keypoints[keypoint_index * 3] += amount
        elif direction == 'y':
            keypoints[keypoint_index * 3 + 1] += amount
        else:
            raise ValueError("Invalid direction specified. Use 'x' or 'y'.")
        
        # Update the 'keypoints' field in the annotation
        annotation['keypoints'] = keypoints
    
    # Save the modified data to a new JSON file
    with open(output_file_path, 'w') as f:
        json.dump(modified_data, f, indent=4)

# Example usage
file_path="./data/datasets/coco/annotations/person_keypoints_val2017.json"
output_file_path = './data/datasets/coco/annotations/person_keypoints_val2017_shifted.json'
move_keypoint(file_path, output_file_path, 'left_shoulder', 'x', 10)
