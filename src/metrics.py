import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, 
    precision_score, recall_score, hamming_loss, 
    jaccard_score, average_precision_score, confusion_matrix
)

class ChestXrayMetrics:
    def __init__(self, target_labels, threshold=0.5):
        self.target_labels = target_labels
        if isinstance(threshold, list):
            self.threshold = np.array(threshold)
        else:
            self.threshold = threshold
        
    def calculate_metrics(self, y_true, y_pred):
        y_pred_bin = (y_pred>self.threshold).astype(int)
        metrics = {}

        metrics['roc_auc_macro'] = self._safe_metric(roc_auc_score, y_true, y_pred, average='macro')
        metrics['pr_auc_macro'] = self._safe_metric(average_precision_score, y_true, y_pred, average='macro')

        metrics['f1_macro'] = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
        metrics['recall_sensitivity_macro'] = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)

        metrics['hamming_loss'] = hamming_loss(y_true, y_pred_bin)
        metrics['jaccard_score_macro'] = jaccard_score(y_true, y_pred_bin, average='macro', zero_division=0)


        specificities = []
        for i in range(len(self.target_labels)):
            tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_bin[:, i], labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
            metrics[f'specificity_{self.target_labels[i]}'] = spec
        metrics['specificity_macro'] = np.mean(specificities)

        metrics['exact_match_acc'] = accuracy_score(y_true, y_pred_bin)
        metrics['element_wise_acc'] = (y_true == y_pred_bin).mean()

        return metrics
    
    def _safe_metric(self, func, y_true, y_score, **kwargs):
        try:
            return func(y_true, y_score, **kwargs)
        except ValueError:
            return 0.0
        
    def get_summary_string(self, train_loss, val_loss, metrics):
        summary = (
            f"\n--- LOSS | Train: {train_loss:.4f} | Val: {val_loss:.4f}\n"
            f"--- MAIN | ROC-AUC: {metrics['roc_auc_macro']:.4f} | PR-AUC (mAP): {metrics['pr_auc_macro']:.4f}\n"
            f"--- CLINICAL | Sensitivity: {metrics['recall_sensitivity_macro']:.4f} | Specificity: {metrics['specificity_macro']:.4f}\n"
            f"--- OTHER | F1: {metrics['f1_macro']:.4f} | Hamming Loss: {metrics['hamming_loss']:.4f}\n"
            f"--- ACCURACY | Exact: {metrics['exact_match_acc']:.4f} | Elem-wise: {metrics['element_wise_acc']:.4f} | Macro Jaccard Score: {metrics['jaccard_score_macro']}"
        )
        return summary

