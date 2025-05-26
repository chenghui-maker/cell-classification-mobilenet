import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===> device: {}".format(device))


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_model(weights_path):
    model = mobilenet_v3_small(pretrained=False)
    in_features = model.classifier[0].out_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, transform, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output.squeeze(), dim=0).cpu().numpy()
        pred_class = int(probs.argmax())
        return pred_class, probs


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--img", required=True, help="Path to the input image")
    parser.add_argument("--weights", required=True, help="Path to the model weights (.pth)")
    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise FileNotFoundError(f"Image not found: {args.img}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    print("===> Loading model")
    model = load_model(args.weights)
    transform = get_val_transform()

    print("===> Predicting")
    pred, probs = predict(model, transform, args.img)
    print(f"Prediction: {pred} (class index)")
    print(f"Probability: class_0 = {probs[0]:.3%}, class_1 = {probs[1]:.3%}")

if __name__ == "__main__":
    main()
