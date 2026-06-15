"""
Evaluation Module for Deepfake Audio Detection
Implements EER, F1 score, confusion matrix, and other metrics.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List, Optional
import os


def calculate_eer(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    Calculate Equal Error Rate (EER).
    
    The EER is the point where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR).
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities
        
    Returns:
        Equal Error Rate (float between 0 and 1)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Calculate FRR (1 - TPR)
    frr = 1 - tpr
    
    # Find the intersection point where FAR = FRR
    # Interpolate to find the exact point
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        # If brentq fails, use the closest point
        eer = fpr[np.nanargmin(np.abs(fpr - frr))]
    
    return eer


def calculate_eer_from_scores(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate EER with threshold.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Prediction scores
        
    Returns:
        Tuple of (EER, threshold, eer_point)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the point where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    
    return eer, eer_threshold, auc(fpr, tpr)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    class_names: List[str] = ['Genuine', 'Deepfake']
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (for EER and AUC)
        class_names: Names of classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Overall Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 Score
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None).tolist()
    
    # Precision and Recall
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    metrics['per_class_accuracy'] = per_class_acc.tolist()
    metrics['per_class_accuracy_dict'] = {
        name: float(acc) for name, acc in zip(class_names, per_class_acc)
    }
    
    # EER (if scores provided)
    if y_scores is not None:
        metrics['eer'] = calculate_eer(y_true, y_scores)
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
    
    # Confusion Matrix
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def calculate_eer_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Find optimal threshold for EER.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        
    Returns:
        Tuple of (optimal_threshold, all_thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find threshold where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    optimal_threshold = thresholds[eer_idx]
    
    return optimal_threshold, thresholds


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Genuine', 'Deepfake'],
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save figure
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix_percentage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Genuine', 'Deepfake'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix with percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_pct = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add counts as secondary annotation
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'(n={cm_pct[i, j]})',
                    ha='center', va='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'ROC Curve'
) -> Dict[str, float]:
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Dictionary with FPR, TPR, thresholds, and AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}


def plot_det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot Detection Error Tradeoff (DET) curve.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, color='red', lw=2)
    plt.plot([0, 1], [1, 0], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('False Negative Rate', fontsize=12)
    plt.title('DET Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['loss'], label='Train', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_performance_report(
    metrics: Dict[str, float],
    class_names: List[str] = ['Genuine', 'Deepfake']
) -> str:
    """
    Generate a formatted performance report.
    
    Args:
        metrics: Dictionary of evaluation metrics
        class_names: Class names
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("DEEPFAKE AUDIO DETECTION - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall metrics
    report.append("OVERALL METRICS:")
    report.append("-" * 40)
    report.append(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"  F1 Score (macro):   {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    report.append(f"  F1 Score (weighted):{metrics['f1_weighted']:.4f}")
    report.append(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    report.append(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    
    if 'eer' in metrics:
        report.append(f"  EER:                {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    if 'roc_auc' in metrics:
        report.append(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    
    report.append("")
    
    # Per-class metrics
    report.append("PER-CLASS METRICS:")
    report.append("-" * 40)
    for i, name in enumerate(class_names):
        acc = metrics['per_class_accuracy'][i]
        f1 = metrics['f1_per_class'][i]
        report.append(f"  {name}:")
        report.append(f"    Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        report.append(f"    F1 Score: {f1:.4f}")
    
    report.append("")
    
    # Confusion Matrix
    report.append("CONFUSION MATRIX:")
    report.append("-" * 40)
    cm = np.array(metrics['confusion_matrix'])
    report.append(f"  {'':>12} {'Predicted':>12}")
    report.append(f"  {'':>12} {class_names[0]:>12} {class_names[1]:>12}")
    report.append(f"  {'True':>12} {class_names[0]:>12}")
    report.append(f"  {'':>12} {cm[0, 0]:>12} {cm[0, 1]:>12}")
    report.append(f"  {'':>12} {class_names[1]:>12}")
    report.append(f"  {'':>12} {cm[1, 0]:>12} {cm[1, 1]:>12}")
    
    report.append("")
    report.append("=" * 60)
    
    # Threshold verification
    report.append("THRESHOLD VERIFICATION:")
    report.append("-" * 40)
    
    acc_met = metrics['accuracy'] >= 0.80
    f1_met = metrics['f1_macro'] >= 0.80
    eer_met = metrics.get('eer', 1) <= 0.12
    per_class_met = all(acc >= 0.75 for acc in metrics['per_class_accuracy'])
    
    report.append(f"  Overall Accuracy ≥ 80%:    {'✓ PASS' if acc_met else '✗ FAIL'} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"  F1 Score ≥ 80%:            {'✓ PASS' if f1_met else '✗ FAIL'} ({metrics['f1_macro']*100:.2f}%)")
    report.append(f"  EER ≤ 12%:                 {'✓ PASS' if eer_met else '✗ FAIL'} ({metrics.get('eer', 1)*100:.2f}%)")
    report.append(f"  Per-Class Accuracy ≥ 75%:  {'✓ PASS' if per_class_met else '✗ FAIL'}")
    
    if acc_met and f1_met and eer_met and per_class_met:
        report.append("")
        report.append("  ✓ ALL THRESHOLDS MET - SUBMISSION VALID")
    else:
        report.append("")
        report.append("  ✗ SOME THRESHOLDS NOT MET - NEEDS IMPROVEMENT")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def save_metrics_to_file(
    metrics: Dict[str, float],
    save_path: str,
    class_names: List[str] = ['Genuine', 'Deepfake']
) -> None:
    """
    Save evaluation metrics to file.
    
    Args:
        metrics: Evaluation metrics
        save_path: Path to save file
        class_names: Class names
    """
    report = generate_performance_report(metrics, class_names)
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Performance report saved to: {save_path}")
