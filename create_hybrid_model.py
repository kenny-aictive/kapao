import torch

def load_model(path):
    try:
        checkpoint = torch.load(path)
        model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Loaded object is not a PyTorch model.")
        return model,checkpoint
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the model: {e}")

def main():
    # Paths to your saved .pt files
    yolo_path = "yolov5s6.pt"
    kapao_path = "kapao_s_coco.pt"

    # Load models
    yolo_model,yolo_checkpoint = load_model(yolo_path)
    kapao_model,kapao_checkpoint = load_model(kapao_path)

    # Extract state dictionaries
    yolo_state_dict = yolo_model.state_dict()
    kapao_state_dict = kapao_model.state_dict()

    # Create a new state_dict by copying layers from kapao_model up to index 33
    new_state_dict = {}
    for name in yolo_state_dict.keys():
        if "33" in name:
            # Use layers from yolo_model for indices after 33
            new_state_dict[name] = yolo_state_dict[name]
        else:
            # Use layers from kapao_model for indices up to 33
            if name in kapao_state_dict:
                new_state_dict[name] = kapao_state_dict[name]

    # Update the yolo_model with the new_state_dict
    yolo_model.load_state_dict(new_state_dict)
    kapao_checkpoint['model']=yolo_model

    # Save the new combined model
    torch.save(yolo_checkpoint, "combined_model.pt")

if __name__ == "__main__":
    main()
