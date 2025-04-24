import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Create output directory for comparison figures
os.makedirs('figures/comparison', exist_ok=True)

def load_metrics():
    """
    Load metrics from saved JSON files for each model.
    
    Verifies that all required metric files exist and can be loaded correctly.
    
    Returns:
        dict: Dictionary of metrics for each model
    """
    models = ['alexnet', 'vgg16', 'vgg16_bn', 'vgg8']
    metrics = {}
    missing_files = []
    
    for model_name in models:
        try:
            # Try to load history from JSON
            with open(f'{model_name}_history.json', 'r') as f:
                metrics[model_name] = json.load(f)
            print(f"Loaded metrics for {model_name}")
        except FileNotFoundError:
            missing_files.append(f"{model_name}_history.json")
    
    # Check if any files were missing
    if missing_files:
        print("\nERROR: The following metric files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the corresponding training scripts first:")
        print("  - python alexnet_cifar10.py")
        print("  - python vgg_cifar10.py")
        print("  - python vgg_bn_cifar10.py")
        print("  - python vgg_reduced_cifar10.py")
        sys.exit(1)
    
    return metrics

def compare_training_curves(metrics):
    """
    Create comparative plots of validation accuracy and loss curves.
    
    Also generates a bar chart of average training time per epoch.
    
    Args:
        metrics (dict): Dictionary containing metrics for each model
    """
    # Create figure for accuracy comparison
    plt.figure(figsize=(12, 6))
    for model_name, model_metrics in metrics.items():
        if 'train_acc' in model_metrics and 'val_acc' in model_metrics:
            plt.plot(model_metrics['val_acc'], label=f'{model_name} Val Acc')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/comparison/accuracy_comparison.png')
    plt.close()
    
    # Create figure for loss comparison
    plt.figure(figsize=(12, 6))
    for model_name, model_metrics in metrics.items():
        if 'train_loss' in model_metrics and 'val_loss' in model_metrics:
            plt.plot(model_metrics['val_loss'], label=f'{model_name} Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/comparison/loss_comparison.png')
    plt.close()
    
    # Create figure for training speed comparison
    plt.figure(figsize=(12, 6))
    avg_times = []
    model_names = []
    
    for model_name, model_metrics in metrics.items():
        if 'time_per_epoch' in model_metrics:
            avg_time = sum(model_metrics['time_per_epoch']) / len(model_metrics['time_per_epoch'])
            avg_times.append(avg_time)
            model_names.append(model_name)
    
    plt.bar(model_names, avg_times)
    plt.xlabel('Model')
    plt.ylabel('Time (s)')
    plt.title('Average Time per Epoch')
    plt.grid(True, axis='y')
    plt.savefig('figures/comparison/time_comparison.png')
    plt.close()

def compare_test_accuracy(metrics):
    """
    Create a bar chart comparing test accuracy across models.
    
    Args:
        metrics (dict): Dictionary containing metrics for each model
    """
    plt.figure(figsize=(10, 6))
    test_accs = []
    model_names = []
    
    for model_name, model_metrics in metrics.items():
        if 'test_acc' in model_metrics:
            test_accs.append(model_metrics['test_acc'])
            model_names.append(model_name)
    
    # Create bars
    bars = plt.bar(model_names, test_accs)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    plt.grid(True, axis='y')
    plt.savefig('figures/comparison/test_accuracy_comparison.png')
    plt.close()

def compare_parameters(metrics):
    """
    Create a bar chart comparing parameter counts across models.
    
    Args:
        metrics (dict): Dictionary containing metrics for each model
    """
    plt.figure(figsize=(10, 6))
    params = []
    model_names = []
    
    for model_name, model_metrics in metrics.items():
        if 'params' in model_metrics:
            params.append(model_metrics['params'] / 1_000_000)  # Convert to millions
            model_names.append(model_name)
    
    # Create bars
    bars = plt.bar(model_names, params)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}M', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Parameters (millions)')
    plt.title('Model Size Comparison')
    plt.grid(True, axis='y')
    plt.savefig('figures/comparison/parameter_comparison.png')
    plt.close()

def compare_performance_vs_size(metrics):
    """
    Create a scatter plot of test accuracy vs. parameter count.
    
    This visualization highlights the efficiency of different models
    in terms of performance relative to model size.
    
    Args:
        metrics (dict): Dictionary containing metrics for each model
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    sizes = []  # Parameter count in millions
    accuracies = []  # Test accuracy
    model_names = []  # For annotations
    
    for model_name, model_metrics in metrics.items():
        if 'params' in model_metrics and 'test_acc' in model_metrics:
            sizes.append(model_metrics['params'] / 1_000_000)  # Convert to millions
            accuracies.append(model_metrics['test_acc'])
            model_names.append(model_name)
    
    # Create scatter plot
    plt.scatter(sizes, accuracies, s=100)
    
    # Add annotations for each point
    for i, model_name in enumerate(model_names):
        plt.annotate(model_name, (sizes[i], accuracies[i]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.xlabel('Parameters (millions)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Performance vs. Model Size')
    plt.grid(True)
    plt.savefig('figures/comparison/performance_vs_size.png')
    plt.close()

def create_summary_table(metrics):
    """
    Create a summary table of key metrics for all models.
    
    The table includes parameter count, test accuracy, training time,
    validation accuracy, and validation loss.
    
    Args:
        metrics (dict): Dictionary containing metrics for each model
        
    Returns:
        DataFrame: Pandas DataFrame containing the summary table
    """
    # Extract data for table
    summary_data = {
        'Model': [],
        'Parameters (M)': [],
        'Test Accuracy (%)': [],
        'Avg Time/Epoch (s)': [],
        'Max Val Accuracy (%)': [],
        'Final Val Loss': []
    }
    
    for model_name, model_metrics in metrics.items():
        summary_data['Model'].append(model_name)
        
        params = model_metrics.get('params', 0) / 1_000_000
        summary_data['Parameters (M)'].append(f"{params:.2f}")
        
        test_acc = model_metrics.get('test_acc', 0)
        summary_data['Test Accuracy (%)'].append(f"{test_acc:.2f}")
        
        if 'time_per_epoch' in model_metrics:
            avg_time = sum(model_metrics['time_per_epoch']) / len(model_metrics['time_per_epoch'])
            summary_data['Avg Time/Epoch (s)'].append(f"{avg_time:.2f}")
        else:
            summary_data['Avg Time/Epoch (s)'].append("N/A")
        
        if 'val_acc' in model_metrics:
            max_val_acc = max(model_metrics['val_acc'])
            summary_data['Max Val Accuracy (%)'].append(f"{max_val_acc:.2f}")
        else:
            summary_data['Max Val Accuracy (%)'].append("N/A")
        
        if 'val_loss' in model_metrics:
            final_val_loss = model_metrics['val_loss'][-1]
            summary_data['Final Val Loss'].append(f"{final_val_loss:.4f}")
        else:
            summary_data['Final Val Loss'].append("N/A")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv('model_comparison_summary.csv', index=False)
    print("Summary table saved to model_comparison_summary.csv")
    
    # Print summary to console
    print("\nModel Comparison Summary:")
    print(df.to_string(index=False))
    
    return df

def compare_overfitting(metrics):
    """
    Analyze and visualize overfitting behavior across models.
    
    Creates plots showing the gap between training and validation
    accuracy over epochs for each model.
    
    Args:
        metrics (dict): Dictionary containing metrics for each model
    """
    plt.figure(figsize=(12, 8))
    
    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        if 'train_acc' in model_metrics and 'val_acc' in model_metrics:
            # Calculate the gap between training and validation accuracy
            train_acc = np.array(model_metrics['train_acc'])
            val_acc = np.array(model_metrics['val_acc'])
            gap = train_acc - val_acc
            
            plt.subplot(2, 2, i+1)
            plt.plot(gap, 'r-', label='Train-Val Gap')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy Gap (%)')
            plt.title(f'{model_name} Overfitting Analysis')
            plt.grid(True)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('figures/comparison/overfitting_comparison.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    print("Loading model metrics...")
    metrics = load_metrics()
    
    print("Generating comparison visualizations...")
    compare_training_curves(metrics)
    compare_test_accuracy(metrics)
    compare_parameters(metrics)
    compare_performance_vs_size(metrics)
    compare_overfitting(metrics)
    
    print("Creating summary table...")
    create_summary_table(metrics)
    
    print("Comparisons complete!")