import torch
import torch.nn as nn


def compare_models(model1, model2):
    model1_layers = dict(model1.named_parameters())
    model2_layers = dict(model2.named_parameters())

    # Compare layer names and dimensions
    model1_keys = set(model1_layers.keys())
    model2_keys = set(model2_layers.keys())
    print(model1_keys)
    print(model2_keys)

    if model1_keys != model2_keys:
        print("The models have different architectures.")
        print("Only present in model 1:", model1_keys - model2_keys)
        print("Only present in model 2:", model2_keys - model1_keys)
        return

    print("The models have the same architecture.")

    # Compare weights
    same_weights = True
    for layer_name, model1_weight in model1_layers.items():
        model2_weight = model2_layers[layer_name]

        if not torch.allclose(model1_weight, model2_weight, atol=1e-7):
            print(f"Weights differ in layer {layer_name}.")
            same_weights = False

    if same_weights:
        print("The models have identical weights.")
    else:
        print("The models have different weights.")


def load_and_compare_models(model1_path, model2_path):
    try:
        checkpoint1 = torch.load(model1_path)
        checkpoint2 = torch.load(model2_path)

        model1 = checkpoint1['model'] if 'model' in checkpoint1 else checkpoint1
        model2 = checkpoint2['model'] if 'model' in checkpoint2 else checkpoint2

        if isinstance(model1, nn.Module) and isinstance(model2, nn.Module):
            compare_models(model1, model2)
        else:
            print("One or both of the loaded objects are not PyTorch models.")
    except Exception as e:
        print(f"An error occurred while loading the models: {e}")


if __name__ == "__main__":
    model1_path = "yolov5s6.pt"
    model2_path = "kapao_s_coco.pt"

    load_and_compare_models(model1_path, model2_path)
