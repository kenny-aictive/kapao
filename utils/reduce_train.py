import json


def create_small_dataset(input_file_path, output_file_path):
    # Read the original JSON file
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    # Calculate the number of entries to keep for the smaller dataset (approximately one-fifth)
    num_images_small = len(data['images']) // 100

    # Create a set of image IDs present in the 'images' section
    image_ids = set(image['id'] for image in data['images'][:num_images_small])

    # Filter the annotations to only include those that have a corresponding image ID in the smaller dataset
    filtered_annotations = [
        annotation for annotation in data['annotations']
        if annotation['image_id'] in image_ids
    ]

    # Create the refined smaller dataset
    data_refined_small = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': data['images'][:num_images_small],
        'annotations': filtered_annotations
    }

    # Save the refined smaller dataset to a new JSON file
    with open(output_file_path, 'w') as f:
        json.dump(data_refined_small, f)


# Paths to the original and output JSON files (You can modify these paths as needed)
input_file_path = "./data/datasets/coco/annotations/person_keypoints_val2017_original.json"
output_file_path = "./data/datasets/coco/annotations/person_keypoints_val2017.json"

# Create the smaller dataset
create_small_dataset(input_file_path, output_file_path)
