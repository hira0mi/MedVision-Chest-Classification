import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report
from prepare_data import TARGET_LABELS
from model import ChestXrayModel
from dataset import ChestXrayDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import train
import logging

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_thresholds_and_update_model(model_path, model, val_loader, device):
    logger.info(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    logger.info("Generating predictions to compute thresholds...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
            probs = torch.sigmoid(outputs).float()
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    best_thresholds = []
    logger.info("\n --- COMPUTING OPTIMAL THRESHOLDS ---")
    
    for i, label in enumerate(TARGET_LABELS):
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        best_idx = np.argmax(f1_scores)
        best_thresh = float(thresholds[best_idx])
        best_thresholds.append(best_thresh)
        
        logger.info(f"{label}: Optimal threshold =  {best_thresh:.4f} (Expected F1 = {f1_scores[best_idx]:.4f})")
    

    model.thresholds = torch.tensor(best_thresholds, dtype=torch.float32, device=device)
    
    new_model_path = 'best_model_with_thresholds.pth'
    torch.save(model.state_dict(), new_model_path)
    logger.info(f"\nModel with embedded thresholds saved as: {new_model_path}")

    logger.info("\n" + "="*60)
    logger.info("FINAL CLASSIFICATION REPORT")
    logger.info("="*60)
    thresh_array = np.array(best_thresholds)
    preds_bin = (all_preds > thresh_array).astype(int)
    
    report = classification_report(all_labels, preds_bin, target_names=TARGET_LABELS, zero_division=0)
    logger.info(report)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = ChestXrayModel(num_classes=len(TARGET_LABELS)).to(device)
    
    transform_val = transforms.Compose([
        train.LungCropping(margin_pct=0.04),
        train.PadToSquare(),
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    
    val_ds = ChestXrayDataset(os.path.join(train.DATA_DIR, 'val.csv'), transform=transform_val)
    val_loader = DataLoader(val_ds, batch_size=train.BATCH_SIZE, shuffle=False, num_workers=4)
    
    find_thresholds_and_update_model(
        model_path='best_model.pth',
        model=model,
        device=device,
        val_loader=val_loader
    )