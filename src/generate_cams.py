import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from prepare_data import TARGET_LABELS
from model import ChestXrayModel
from dataset import ChestXrayDataset
import train 

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def find_true_positive(val_dataset, model, target_class_idx, device):
   
    model.eval()
    threshold = model.thresholds[target_class_idx].item()
    
    print(f"Searching for a True Positive sample for: {TARGET_LABELS[target_class_idx]} (Threshold: {threshold:.2f})...")
    
    for i in range(len(val_dataset)):
        image, labels = val_dataset[i]
        
        if labels[target_class_idx] != 1.0:
            continue
            
        input_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).squeeze()[target_class_idx].item()

        if prob > threshold:
            print(f"Found! Image index: {i}, Probability: {prob:.2f}")
            return image, labels, prob, i
            
    print("Could not find a True Positive sample for this class")
    return None, None, None, None


def save_gradcam(image, prob, target_class_idx, model, device, save_name):
    input_tensor = image.unsqueeze(0).to(device)
    top_class_name = TARGET_LABELS[target_class_idx]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_img = image.permute(1, 2, 0).cpu().numpy()
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1)

    target_layers = [model.backbone.features[-1]]
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("original")
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title(f"True Positive: {top_class_name}\nPredicted Prob: {prob:.2f} (Actual: 1)", color='green')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()
    print(f"Saved: {save_name}\n")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ChestXrayModel(num_classes=len(TARGET_LABELS))
    model.load_state_dict(torch.load('best_model_with_thresholds.pth', map_location=device, weights_only=True))
    model.to(device)
    
    transform_val = transforms.Compose([
        train.LungCropping(margin_pct=0.04),
        train.PadToSquare(),
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    val_ds = ChestXrayDataset(os.path.join(train.DATA_DIR, 'val.csv'), transform=transform_val)
    
    os.makedirs('showcase_images', exist_ok=True)
    

    for path_name in TARGET_LABELS:
        idx = TARGET_LABELS.index(path_name)
        image, labels, prob, sample_idx = find_true_positive(val_ds, model, idx, device)
        
        if image is not None:
            save_name = f"showcase_images/TruePositive_{path_name}.png"
            save_gradcam(image, prob, idx, model, device, save_name)