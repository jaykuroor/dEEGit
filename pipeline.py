"""
dEEGit Pipeline: End-to-end EEG digit classification using EEGNet.

This script provides a complete pipeline to:
1. Load and preprocess EEG data
2. Train an EEGNet model
3. Evaluate model performance
4. Visualize results

Usage:
    python pipeline.py --mode train      # Train a new model
    python pipeline.py --mode eval       # Evaluate existing model
    python pipeline.py --mode both       # Train and evaluate
    python pipeline.py --mode demo       # Quick demo with fewer epochs
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.dataset import load_data
from src.model import EEGNet, EEGNetBetter
from src.train import train_model, load_checkpoint
from src.eval import (
    evaluate_model, 
    print_evaluation_report, 
    plot_confusion_matrix,
    plot_training_history,
    get_misclassified_samples,
    print_misclassified_summary
)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_dataloaders(
    train_dataset, 
    val_dataset, 
    test_dataset, 
    batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, val, and test datasets."""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader


def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = None,
    learning_rate: float = None,
    save_path: str = "checkpoints"
) -> dict:
    """Run the training pipeline."""
    if epochs is None:
        epochs = config.EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_path,
        patience=15,
        scheduler_patience=5
    )
    
    return history


def run_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str = None,
    save_plots: bool = True,
    plots_path: str = "plots"
) -> dict:
    """Run the evaluation pipeline."""
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, device)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print report
    print_evaluation_report(results)
    
    # Get and print misclassified samples
    misclassified = get_misclassified_samples(results, num_samples=10)
    print_misclassified_summary(misclassified)
    
    # Plot confusion matrix
    if save_plots:
        os.makedirs(plots_path, exist_ok=True)
        plot_confusion_matrix(
            results, 
            save_path=os.path.join(plots_path, 'confusion_matrix.png'),
            normalize=True
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='dEEGit: EEG-Based Digit Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline.py --mode train
    python pipeline.py --mode eval --checkpoint checkpoints/best_model.pth
    python pipeline.py --mode both --epochs 50
    python pipeline.py --mode demo
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='both',
        choices=['train', 'eval', 'both', 'demo'],
        help='Pipeline mode: train, eval, both, or demo (default: both)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='eegnet',
        choices=['eegnet', 'eegnet_seq'],
        help='Model architecture: eegnet or eegnet_seq (default: eegnet)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of training epochs (default: {config.EPOCHS})'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help=f'Learning rate (default: {config.LEARNING_RATE})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint for evaluation'
    )
    parser.add_argument(
        '--resample_method',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Resampling method: 0=FFT, 1=Polyphase (default), 2=CubicSpline'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable saving plots'
    )
    
    args = parser.parse_args()
    
    # Demo mode settings
    if args.mode == 'demo':
        args.epochs = 10
        print("\n*** DEMO MODE: Running with 10 epochs ***\n")
    
    # Get device
    device = get_device()
    
    # Load data
    print("\nLoading and preprocessing data...")
    batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE
    
    train_dataset, val_dataset, test_dataset = load_data(
        resample_method=args.resample_method
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'eegnet':
        model = EEGNet(
            chan_num=config.CHAN_NUM,
            time_len=config.TARGET_LEN,
            num_classes=10
        )
    else:
        model = EEGNetBetter(
            chan_num=config.CHAN_NUM,
            time_len=config.TARGET_LEN,
            num_classes=10
        )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    history = None
    
    # Training
    if args.mode in ['train', 'both', 'demo']:
        epochs = args.epochs if args.epochs else config.EPOCHS
        lr = args.lr if args.lr else config.LEARNING_RATE
        
        history = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=lr
        )
        
        # Plot training history
        if not args.no_plots and history:
            os.makedirs('plots', exist_ok=True)
            plot_training_history(
                history,
                save_path='plots/training_history.png'
            )
    
    # Evaluation
    if args.mode in ['eval', 'both', 'demo']:
        checkpoint_path = args.checkpoint if args.mode == 'eval' else 'checkpoints/best_model.pth'
        
        results = run_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            checkpoint_path=checkpoint_path,
            save_plots=not args.no_plots
        )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
