"""
Evaluation metrics for methylation prediction.

Provides functions to compute correlation and classification metrics,
as well as training-time metrics computation.
"""
from typing import Dict, Optional, List
import numpy as np
import scipy
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve


# =============================================================================
# Classification Metrics
# =============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with classification metrics (accuracy, precision, recall, f1, auc, auprc)
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred > threshold, labels=[0, 1]
    ).ravel()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    
    eps = 1e-20  # Small value to avoid division by zero
    
    result = {
        'total_labels': len(y_true),
        'positive_labels': np.count_nonzero(y_true == 1),
        'negative_labels': np.count_nonzero(y_true == 0),
        'accuracy': (tp + tn) / (tn + fp + fn + tp + eps),
        'precision': tp / (tp + fp + eps),
        'recall': tp / (tp + fn + eps),
        'sensitivity': tp / (tp + fn + eps),
        'specificity': tn / (fp + tn + eps),
    }
    result['f1_score'] = 2 * (result['precision'] * result['recall']) / \
                         (result['precision'] + result['recall'] + eps)
    result['auc'] = auc(fpr, tpr)
    result['auprc'] = auc(recalls, precisions)
    
    # Round float values
    result = {
        k: round(float(v), 4) if isinstance(v, (np.floating, float)) else v
        for k, v in result.items()
    }
    
    return result


def find_opt_thres_ROC(fpr, tpr, thresholds):
    """Find optimal threshold using Youden's index on ROC curve."""
    y = tpr - fpr
    youden_index = np.argmax(y)
    return thresholds[youden_index]


def find_opt_thres_PR(precisions, recalls, thresholds):
    """Find optimal threshold using max(precision + recall) on PR curve."""
    y = precisions + recalls
    youden_index = np.argmax(y)
    return thresholds[youden_index]


def get_metrics(y_reals: np.ndarray, y_preds: np.ndarray, thres: float = 0.5) -> Dict:
    """
    Calculate comprehensive metrics for binary classification.
    
    Includes optimal threshold finding.
    
    Args:
        y_reals: Ground truth labels
        y_preds: Predicted probabilities
        thres: Classification threshold
        
    Returns:
        Dictionary containing all metrics including opt_thres_ROC and opt_thres_PR
    """
    tn, fp, fn, tp = confusion_matrix(y_reals, y_preds > thres, labels=[0, 1]).ravel()
    fpr, tpr, thresholds = roc_curve(y_reals, y_preds)
    opt_thres1 = find_opt_thres_ROC(fpr, tpr, thresholds)
    precisions, recalls, thresholds = precision_recall_curve(y_reals, y_preds)
    opt_thres2 = find_opt_thres_PR(precisions, recalls, thresholds)
    
    result = {
        'All labels': len(y_reals),
        'Positive': np.count_nonzero(y_reals == 1),
        'Negative': np.count_nonzero(y_reals == 0),
        'Accuracy': (tp + tn) / (tn + fp + fn + tp + 1e-20),
        'Precision': tp / (tp + fp + 1e-20),
        'Recall': tp / (tp + fn + 1e-20),
        'Sensitivity': tp / (tp + fn + 1e-20),
        'Specificity': tn / (fp + tn + 1e-20),
    }
    result['F1score'] = 2 * (result['Precision'] * result['Recall']) / \
                        (result['Precision'] + result['Recall'] + 1e-20)
    result['AUC'] = auc(fpr, tpr)
    result['AUPRC'] = auc(recalls, precisions)
    result['opt_thres_ROC'] = opt_thres1
    result['opt_thres_PR'] = opt_thres2
    result = {x: round(float(result[x]), 4) if isinstance(result[x], np.float64) else result[x] 
              for x in result}
    return result


# =============================================================================
# Correlation Metrics
# =============================================================================

def compute_correlation_metrics(a1: np.ndarray, a2: np.ndarray) -> Dict[str, float]:
    """
    Compute correlation metrics between two arrays.
    
    Args:
        a1: First array (predictions)
        a2: Second array (ground truth)
        
    Returns:
        Dictionary with correlation metrics (pearson, rsquare, spearman, RMSE)
    """
    spearmanr = scipy.stats.spearmanr(a1, a2).statistic
    r = np.corrcoef(a1, a2)[0][1]
    r2 = r * r
    rmse = np.sqrt(np.mean((a1 - a2) ** 2))
    
    return {
        'sites': len(a1),
        'pearson': r,
        'rsquare': r2,
        'spearman': spearmanr,
        'RMSE': rmse
    }


# =============================================================================
# Training Metrics
# =============================================================================

def compute_metrics(pred, tokenizer: Dict, methy_types: Optional[List[str]] = None):
    """
    Compute metrics for each methylation type during training.
    
    Args:
        pred: Prediction object with predictions, label_ids, and inputs
        tokenizer: Tokenizer dictionary
        methy_types: List of methylation types to evaluate
        
    Returns:
        Dictionary of metrics per methylation type
    """
    if methy_types is None:
        methy_types = ['[CpG]', '[CHG]', '[CHH]', '[m6A]']
    
    predictions, labels, decoder_input_ids = pred.predictions, pred.label_ids, pred.inputs
    positive, negative = tokenizer['+'], tokenizer['-']
    res = {}
    label_mask = (labels == positive) | (labels == negative)
    
    for meth_type in methy_types:
        meth_id = tokenizer[meth_type]
        mask_type = (decoder_input_ids == meth_id)
        mask = mask_type & label_mask
        labels_mask, predictions_mask = labels[mask], predictions[mask]
        predictions_mask = np.exp(predictions_mask[:, positive]) / np.sum(
            np.exp(predictions_mask[:, [positive, negative]]), axis=1)
        labels_mask = 11 - labels_mask
        
        if len(labels_mask) == 0:
            continue
        
        metrics = get_metrics(
            y_reals=labels_mask,
            y_preds=predictions_mask
        )
        res[meth_type] = metrics
    
    return res
