import torch
import torch.nn as nn
import torch.optim as optim
from model import DAVE2
from dataset import SelfDrivingDataset
from torch.utils.data import DataLoader

def train_model():
    CATALOG_FILE = "../data/catalog_0.catalog"
    IMAGE_DIR = ""
    dataset = SelfDrivingDataset(CATALOG_FILE, IMAGE_DIR)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAVE2().to(device)               # Bug 4 fixed: DAVE2 → DAVE2()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()      # Bug 5 fixed: loss.item → loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{(epoch+1)/epochs}], Loss: {avg_loss:.4f}")  # Bug 6 fixed: parens added

    torch.save(model.state_dict(), "dave2_robot_model.pth")
    print("TRAINING COMPLETE! MODEL SAVED!")

if __name__ == "__main__":
    train_model()