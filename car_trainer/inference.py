import torch
import torchvision.transforms as transforms
from PIL import Image
from model import DAVE2

def load_model(weights_path: str, device: torch.device) -> DAVE2:
    model = DAVE2().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path: str, image_size: tuple = (66, 200)) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)  # model normalizes internally via (x/127.5)-1
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension → (1, 3, H, W)

def predict_steering(model: DAVE2, image_tensor: torch.Tensor, device: torch.device) -> float:
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output.item()  # returns a float in [-1, 1] (tanh output)

def run_inference(image_path: str, weights_path: str = "dave2_robot_model.pth") -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)
    image_tensor = preprocess_image(image_path)
    steering_angle = predict_steering(model, image_tensor, device)
    print(f"Predicted steering angle: {steering_angle:.4f}  (range: -1.0 to 1.0)")
    return steering_angle

# --- Batch inference (for evaluating a list of images) ---
def run_batch_inference(image_paths: list, weights_path: str = "dave2_robot_model.pth") -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)

    results = []
    for path in image_paths:
        tensor = preprocess_image(path).to(device)
        with torch.no_grad():
            angle = model(tensor).item()
        results.append({"image": path, "steering_angle": angle})
        print(f"{path}  →  {angle:.4f}")

    return results

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [weights_path]")
        print("Example: python inference.py ../data/frame_001.jpg dave2_robot_model.pth")
        sys.exit(1)

    image_path   = sys.argv[1]
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "dave2_robot_model.pth"

    run_inference(image_path, weights_path)