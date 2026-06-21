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
LEARNING_RATE = 0.0001
EPOCHS = 1

class CustomMappedDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.samples = []
        self.transform = transform
        
        # Auto-descend if there is a single subdirectory with expected names
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            if len(subdirs) == 1 and subdirs[0] in ["Background Remove Dataset", "Original Dataset", "Augmented Dataset", os.path.basename(data_dir)]:
                data_dir = os.path.join(data_dir, subdirs[0])
                print(f"Automatically descended into sub-folder: {data_dir}")
                
        def normalize(name):
            return name.lower().replace(" ", "_").replace("___", "_").replace("__", "_").strip()
            
        normalized_class_names = [normalize(c) for c in class_names]
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")
            
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            norm_folder = normalize(folder_name)
            class_idx = -1
            
            # Match 1: Exact or suffix/prefix match
            for idx, c_norm in enumerate(normalized_class_names):
                if norm_folder == c_norm or norm_folder in c_norm or c_norm in norm_folder:
                    class_idx = idx
                    break
                    
            # Match 2: Adding dataset prefix if not found
            if class_idx == -1:
                for idx, c_norm in enumerate(normalized_class_names):
                    if "medicinal_background_remove_dataset_" + norm_folder == c_norm:
                        class_idx = idx
                        break
            
            if class_idx == -1:
                continue
                
            print(f"Mapped folder '{folder_name}' to class index {class_idx}")
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images for training.")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for fine-tuning: {device}")
    
    # Load class names
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Classes file '{CLASSES_PATH}' not found!")
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)
        
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights file '{MODEL_PATH}' not found!")
        
    print("Loading existing model weights...")
    model = models.mobilenet_v2()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    
    # Backup original weights
    if not os.path.exists(BACKUP_PATH):
        print(f"Creating backup of original model weights to '{BACKUP_PATH}'...")
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        
    # Set up fine-tuning parameters
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the classification head & last feature block for fine-tuning
    for param in model.features[18].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Data Augmentation & Normalization
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading and preparing local dataset...")
    dataset = CustomMappedDataset(DATASET_DIR, class_names, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only unfrozen parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    print("\nStarting fine-tuning...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        corrects = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                acc = (corrects.double() / total) * 100
                avg_loss = running_loss / total
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(loader)}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
                
    # Save updated weights
    print(f"\nFine-tuning complete. Saving updated weights to '{MODEL_PATH}'...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Weights saved successfully!")

if __name__ == "__main__":
    main()
