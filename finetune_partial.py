import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import shutil

MODEL_PATH = "agrishield_model.pth"
BACKUP_PATH = "agrishield_model_backup.pth"
CLASSES_PATH = "class_names.json"
DATASET_DIR = "A Multi-Class Medicinal Plant Leaf Dataset with Mu/Background Remove Dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.01  # Slightly higher for quick adaptation of single layer
EPOCHS = 2

class CustomMappedDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.samples = []
        self.transform = transform
        
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            if len(subdirs) == 1 and subdirs[0] in ["Background Remove Dataset", "Original Dataset", "Augmented Dataset", os.path.basename(data_dir)]:
                data_dir = os.path.join(data_dir, subdirs[0])
                
        def normalize(name):
            return name.lower().replace(" ", "_").replace("___", "_").replace("__", "_").strip()
            
        normalized_class_names = [normalize(c) for c in class_names]
        
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            norm_folder = normalize(folder_name)
            class_idx = -1
            
            for idx, c_norm in enumerate(normalized_class_names):
                if norm_folder == c_norm or norm_folder in c_norm or c_norm in norm_folder:
                    class_idx = idx
                    break
                    
            if class_idx == -1:
                for idx, c_norm in enumerate(normalized_class_names):
                    if "medicinal_background_remove_dataset_" + norm_folder == c_norm:
                        class_idx = idx
                        break
            
            if class_idx == -1:
                continue
                
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images for local partial training.")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    device = torch.device("cpu") # CPU is fine since we are only training a single linear layer subset
    
    # Load class names
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)
        
    # Load model
    model = models.mobilenet_v2()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    
    # Backup original weights if not exists
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        
    # Freeze all layers completely
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Data Augmentation & Normalization
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomMappedDataset(DATASET_DIR, class_names, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    # Use standard SGD without momentum to avoid weight leaks on frozen nodes
    optimizer = optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting partial fine-tuning (Optimizing classes 18-39 only)...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        corrects = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Zero out gradients for classes outside 18-39 to freeze their classifier weights
            with torch.no_grad():
                if model.classifier[1].weight.grad is not None:
                    model.classifier[1].weight.grad[0:18] = 0.0
                    model.classifier[1].weight.grad[40:63] = 0.0
                if model.classifier[1].bias.grad is not None:
                    model.classifier[1].bias.grad[0:18] = 0.0
                    model.classifier[1].bias.grad[40:63] = 0.0
            
            optimizer.step()
            
            # Stats
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
        acc = (corrects.double() / total) * 100
        avg_loss = running_loss / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Completed | Loss: {avg_loss:.4f} | Training Acc: {acc:.2f}%")
                
    # Save updated weights
    print(f"\nSaving updated weights to '{MODEL_PATH}'...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Weights saved successfully!")

if __name__ == "__main__":
    main()
