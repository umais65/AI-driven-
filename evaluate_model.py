import os
import json
import argparse
import shutil
import numpy as np
from PIL import Image

# PyTorch Imports
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Metrics Imports
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Plotting Imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

MODEL_PATH = "agrishield_model.pth"
CLASSES_PATH = "class_names.json"
IMAGE_SIZE = 224

def load_classes():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Classes list file '{CLASSES_PATH}' not found!")
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)

def load_model(num_classes, device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights file '{MODEL_PATH}' not found!")
    
    # Initialize MobileNetV2 structure
    model = models.mobilenet_v2()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def create_synthetic_dataset(class_names, temp_dir="temp_synthetic_val"):
    """
    Creates a temporary synthetic validation dataset containing dummy images
    so the evaluation pipeline can be run and verified locally.
    """
    print(f"\n[Self-Test] Generating synthetic dataset in '{temp_dir}'...")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate 1-2 dummy images per class
    np.random.seed(42)
    for class_name in class_names:
        class_path = os.path.join(temp_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        # Create 2 random images
        for i in range(2):
            # Generate a random colored image to simulate a leaf cell structure
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            img_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            img_arr[:, :] = color
            
            # Add some random noise
            noise = np.random.normal(0, 15, img_arr.shape).astype(np.uint8)
            img_arr = np.clip(img_arr + noise, 0, 255)
            
            img = Image.fromarray(img_arr)
            img.save(os.path.join(class_path, f"dummy_{i}.jpg"))
            
    print("[Self-Test] Synthetic dataset generation complete.")

def evaluate(model, data_loader, class_names, device):
    print("\nRunning model inference on validation set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("Inference complete. Calculating metrics...")
    
    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds)
    
    # 2. Precision, Recall, F1
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print("\n=============================================")
    print("           MODEL EVALUATION SUMMARY          ")
    print("=============================================")
    print(f"Global Accuracy:          {acc * 100:.2f}%")
    print(f"Weighted Precision:       {precision_w * 100:.2f}%")
    print(f"Weighted Recall (Sensitivity): {recall_w * 100:.2f}%")
    print(f"Weighted F1-Score:        {f1_w * 100:.2f}%")
    print("---------------------------------------------")
    print(f"Macro Precision:          {precision_m * 100:.2f}%")
    print(f"Macro Recall:             {recall_m * 100:.2f}%")
    print(f"Macro F1-Score:           {f1_m * 100:.2f}%")
    print("=============================================")
    
    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    # Save confusion matrix plot
    if PLOT_AVAILABLE:
        print("\nPlotting confusion matrix...")
        plt.figure(figsize=(26, 26))
        
        # If seaborn is available, use heatmap, otherwise standard imshow
        try:
            sns.heatmap(
                cm, 
                annot=False, 
                cmap="YlGnBu", 
                xticklabels=class_names, 
                yticklabels=class_names,
                square=True,
                cbar_kws={"shrink": .8}
            )
        except Exception:
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar(shrink=0.8)
            plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
            plt.yticks(range(len(class_names)), class_names, fontsize=6)
            
        plt.title("AgriShield Leaf Classification - Confusion Matrix", fontsize=20, pad=20)
        plt.ylabel("True Class Label", fontsize=14)
        plt.xlabel("Predicted Class Label", fontsize=14)
        plt.tight_layout()
        
        plot_name = "confusion_matrix.png"
        plt.savefig(plot_name, dpi=120)
        plt.close()
        print(f"Confusion matrix plot saved successfully as '{plot_name}' in the workspace.")
    else:
        print("\n[Warning] Matplotlib/Seaborn not available. Skipping confusion matrix plotting.")
        
    return acc, precision_w, recall_w, f1_w

class CustomMappedDataset(torch.utils.data.Dataset):
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
            print(f"[Error] Directory '{data_dir}' does not exist.")
            return
            
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
                print(f"[Warning] Could not match folder '{folder_name}' to any trained class. Skipping.")
                continue
                
            print(f"Mapped folder '{folder_name}' to class '{class_names[class_idx]}' (index {class_idx})")
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images across classes.")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    parser = argparse.ArgumentParser(description="AgriShield Model Evaluator")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None, 
        help="Path to validation dataset directory (structured as subfolders per class)"
    )
    args = parser.parse_args()
    
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is required to calculate metrics. Please install it using 'pip install scikit-learn'.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")
    
    # Load class names
    class_names = load_classes()
    print(f"Loaded {len(class_names)} trained classes.")
    
    # Load model
    model = load_model(len(class_names), device)
    
    # Validate data source
    use_temp = False
    data_dir = args.data_dir
    
    if data_dir is None or not os.path.exists(data_dir):
        if data_dir is not None:
            print(f"[Warning] Specified data directory '{data_dir}' not found.")
        print("No validation dataset folder found locally (trained on Colab). Running script in SELF-TEST mode.")
        data_dir = "temp_synthetic_val"
        create_synthetic_dataset(class_names, temp_dir=data_dir)
        use_temp = True
        
    # Setup DataLoader
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        if use_temp:
            dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
        else:
            dataset = CustomMappedDataset(data_dir, class_names, transform=val_transforms)
            
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Run evaluation
        evaluate(model, loader, class_names, device)
        
    finally:
        # Clean up temporary synthetic dataset if used
        if use_temp and os.path.exists(data_dir):
            print("\nCleaning up temporary synthetic dataset folder...")
            shutil.rmtree(data_dir)
            print("Temporary folder cleaned.")

if __name__ == "__main__":
    main()
