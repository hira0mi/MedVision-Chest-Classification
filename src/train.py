import os
import logging
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from prepare_data import TARGET_LABELS
from dataset import ChestXrayDataset
from model import ChestXrayModel
from metrics import ChestXrayMetrics
from lung_cropping import LungCropping


os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

NUM_CLASSES = len(TARGET_LABELS)
BATCH_SIZE = 16


DATA_DIR = '../data'

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True, pos_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weights = pos_weights

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weights)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return TF.pad(image, (p_left, p_top, p_right, p_bottom), fill=0, padding_mode='constant')

class ChestXrayTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, metrics_calc, writer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics_calc = metrics_calc
        self.writer = writer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)

        self.best_pr_auc = 0.0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_roc_auc':[], 'val_roc_auc': [],
            'train_f1': [], 'val_f1':[]
        }
        
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler()

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [],[]

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().float() 
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        train_metrics = self.metrics_calc.calculate_metrics(all_labels, all_preds)
        
        return avg_loss, train_metrics
    def set_optimizer(self, new_optimizer):
        self.optimizer = new_optimizer
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
    def _val_epoch(self):
        self.model.eval()
        all_preds, all_labels =[],[]
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs).float()
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = val_loss / len(self.val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        

        val_metrics = self.metrics_calc.calculate_metrics(all_labels, all_preds)
        
        return avg_loss, val_metrics
    
    def _save_plots(self, current_epoch):
        epochs_range = range(1, current_epoch + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs_range, self.history['val_loss'], label='Val Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'loss_curve.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.history['train_roc_auc'], label='Train ROC-AUC', color='lightblue', linestyle='--')
        plt.plot(epochs_range, self.history['val_roc_auc'], label='Val ROC-AUC', color='blue', marker='s')
        plt.title('Train vs Val ROC-AUC')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'roc_auc_curve.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, self.history['train_f1'], label='Train F1 score', color='lightblue', linestyle='--')
        plt.plot(epochs_range, self.history['val_f1'], label='Val F1 score', color='blue', marker='s')
        plt.title('Train vs Val F1 score')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'f1_score_curve.png'))
        plt.close()

    def _log_gradcam_sample(self, epoch):
        self.model.eval()
        
        idx = random.randint(0, len(self.val_loader.dataset) - 1)
        image, labels = self.val_loader.dataset[idx]
        

        input_tensor = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output).squeeze().cpu().numpy()

        predicted_indices = np.where(probs > self.metrics_calc.threshold)[0]
        if len(predicted_indices) == 0:
            predicted_indices = np.array([int(np.argmax(probs))])

        predicted_labels = [TARGET_LABELS[i] for i in predicted_indices]
        predicted_probs = [probs[i] for i in predicted_indices]

        top_class_idx = int(predicted_indices[np.argmax(predicted_probs)])
        top_class_name = TARGET_LABELS[top_class_idx]
        top_class_prob = probs[top_class_idx]
        actual_label = int(labels[top_class_idx])

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        rgb_img = image.permute(1, 2, 0).cpu().numpy()
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1) 
        

        target_layers =[self.model.backbone.features[-1]]
        targets = [ClassifierOutputTarget(top_class_idx)]
        
        with GradCAM(model=self.model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title("original")
        axes[0].axis('off')
        
        axes[1].imshow(visualization)
        predicted_text = ", ".join(
            f"{label} ({prob:.2f})" for label, prob in zip(predicted_labels, predicted_probs)
        )
        title = (
            f"CAM attention: {top_class_name}\n"
            f"Predicted labels: {predicted_text}\n"
            f"Actual for CAM target: {actual_label}"
        )
        axes[1].set_title(title, color='green' if actual_label == 1 else 'red')
        axes[1].axis('off')
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, f'gradcam_epoch_{epoch}.png')
        plt.savefig(save_path)
        

        self.writer.add_figure('XAI/Validation_Sample', fig, epoch)
        
        plt.close(fig)

    def fit(self, epochs, start_epoch=1, log_gradcam=True, close_writer=True):
        logger.info("=== Training started ===")
        end_epoch = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
        
            train_loss, train_metrics = self._train_epoch()
            val_loss, val_metrics = self._val_epoch()
            logger.info(f"\n========== [Epoch {epoch}/{end_epoch - 1}] ==========")
            logger.info(">>> Train metrics:")
            logger.info(self.metrics_calc.get_summary_string(train_loss, val_loss, train_metrics))
            logger.info(">>> Val metrics:")
            logger.info(self.metrics_calc.get_summary_string(train_loss, val_loss, val_metrics))

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_roc_auc'].append(train_metrics['roc_auc_macro'])
            self.history['val_roc_auc'].append(val_metrics['roc_auc_macro'])
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_f1'].append(val_metrics['f1_macro'])

            self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
            self.writer.add_scalars('ROC-AUC', {'Train': train_metrics['roc_auc_macro'], 'Val': val_metrics['roc_auc_macro']}, epoch)
            self.writer.add_scalars('F1', {'Train': train_metrics['f1_macro'], 'Val': val_metrics['f1_macro']}, epoch)
            
            self._save_plots(epoch)

            if log_gradcam:
                self._log_gradcam_sample(epoch)

            if val_metrics['pr_auc_macro'] > self.best_pr_auc:
                self.best_pr_auc = val_metrics['pr_auc_macro']
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info(f"\n+++ New best model saved (Val pr-auc: {self.best_pr_auc:.4f}) +++\n")
            self.scheduler.step(val_metrics['pr_auc_macro'])
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current Learning Rate: {current_lr}")

        if close_writer:
            self.writer.close()
        logger.info("=== Training completed ===")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    logger.info(f"Using device: {device}")
    
    pos_weights_path = os.path.join(DATA_DIR, 'pos_weights.pt')
    if os.path.exists(pos_weights_path):
        pos_weights = torch.load(pos_weights_path).to(device)
        pos_weights = torch.clamp(pos_weights, max=10.0) 
        criterion = FocalLoss(pos_weights=pos_weights, gamma=2.0)
        logger.info("Using Focal Loss with pos_weights")
    else:
        criterion = FocalLoss(gamma=2.0)
        logger.warning("Weights not found, using default Focal Loss") 

    model = ChestXrayModel(num_classes=NUM_CLASSES).to(device)


    for param in model.backbone.features.parameters():
        param.requires_grad = False

    optimizer1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-3
    )
    

    transform_train = transforms.Compose([
        LungCropping(margin_pct=0.04),
        PadToSquare(),
        transforms.Resize((512,512)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    
    transform_val = transforms.Compose([
        LungCropping(margin_pct=0.04),
        PadToSquare(),
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    logger.info("Creating datasets and dataloaders...")
    train_ds = ChestXrayDataset(os.path.join(DATA_DIR, 'train.csv'), transform=transform_train)
    val_ds = ChestXrayDataset(os.path.join(DATA_DIR, 'val.csv'), transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    metrics_calc = ChestXrayMetrics(TARGET_LABELS)
    writer = SummaryWriter(log_dir='runs/chest_xray_experiment')

    trainer = ChestXrayTrainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer1, 
        device=device,
        metrics_calc=metrics_calc,
        writer=writer
    )
    trainer.fit(3, start_epoch=1, log_gradcam=False, close_writer=False)
    for param in model.backbone.features.parameters():
        param.requires_grad = True

    optimizer2 = optim.Adam([

        {'params': model.backbone.features.parameters(), 'lr': 1e-6},
        {'params': model.backbone.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-3)

    trainer.set_optimizer(optimizer2) 

    trainer.fit(7, start_epoch=4, log_gradcam=True, close_writer=True)

