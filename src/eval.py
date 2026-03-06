"""
Evaluation module for EEGNet model.
Provides comprehensive evaluation metrics, confusion matrix, and visualization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
from typing import Optional


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[list[str]] = None
) -> dict:
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cpu/cuda)
        class_names: Optional list of class names (digits 0-9)
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1_score': f1_per_class,
            'support': support_per_class
        },
        'class_names': class_names
    }
    
    return results


def print_evaluation_report(results: dict) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        results: Dictionary from evaluate_model()
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1_score']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print("-"*60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*60)
    
    class_names = results['class_names']
    per_class = results['per_class']
    
    for i, name in enumerate(class_names):
        if i < len(per_class['precision']):
            print(f"{name:<10} {per_class['precision'][i]:<12.4f} {per_class['recall'][i]:<12.4f} "
                  f"{per_class['f1_score'][i]:<12.4f} {int(per_class['support'][i]):<10}")
    
    print("-"*60)
    print(f"\nTotal samples: {len(results['labels'])}")


def plot_confusion_matrix(
    results: dict, 
    save_path: Optional[str] = None,
    normalize: bool = True,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot and optionally save confusion matrix.
    
    Args:
        results: Dictionary from evaluate_model()
        save_path: Optional path to save the figure
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
    """
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format=fmt)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined loss and accuracy (normalized)
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    l1, = ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    l2, = ax2.plot(epochs, history['val_acc'], 'b-', label='Val Acc', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Accuracy', color='b')
    ax1.set_title('Validation Loss vs Accuracy')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def get_misclassified_samples(
    results: dict,
    num_samples: int = 10
) -> list[dict]:
    """
    Get details of misclassified samples.
    
    Args:
        results: Dictionary from evaluate_model()
        num_samples: Number of misclassified samples to return
        
    Returns:
        List of dictionaries with misclassification details
    """
    predictions = results['predictions']
    labels = results['labels']
    probs = results['probabilities']
    class_names = results['class_names']
    
    misclassified_idx = np.where(predictions != labels)[0]
    
    misclassified = []
    for idx in misclassified_idx[:num_samples]:
        misclassified.append({
            'index': idx,
            'true_label': class_names[labels[idx]],
            'predicted_label': class_names[predictions[idx]],
            'confidence': probs[idx][predictions[idx]],
            'true_class_prob': probs[idx][labels[idx]]
        })
    
    return misclassified


def print_misclassified_summary(misclassified: list[dict]) -> None:
    """
    Print summary of misclassified samples.
    
    Args:
        misclassified: List from get_misclassified_samples()
    """
    print("\nMisclassified Samples Summary:")
    print("-"*70)
    print(f"{'Index':<8} {'True':<8} {'Predicted':<12} {'Confidence':<12} {'True Prob':<12}")
    print("-"*70)
    
    for sample in misclassified:
        print(f"{sample['index']:<8} {sample['true_label']:<8} {sample['predicted_label']:<12} "
              f"{sample['confidence']:<12.4f} {sample['true_class_prob']:<12.4f}")
