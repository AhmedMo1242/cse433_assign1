# Comparative Analysis of CNN Architectures on CIFAR-10
### CSE 433: Emerging Topics - Deep Learning (Spring 2025)

**Student:** Ahmed Hassan (120210113)  
**Submitted to:** Prof. Ehab Elshazly, Eng. Esraa Abdelrazek, Eng. Yasser Rohaim

## Project Overview

This project implements and compares four different CNN architectures on the CIFAR-10 dataset: AlexNet, VGG16, VGG16 with Batch Normalization (VGG16-BN), and a reduced VGG variant (VGG8).

## Project Structure

- `alexnet_cifar10.py` - AlexNet implementation and training
- `vgg_cifar10.py` - VGG16 implementation and training
- `vgg_bn_cifar10.py` - VGG16 with Batch Normalization implementation and training
- `vgg_reduced_cifar10.py` - Reduced VGG (VGG8) implementation and training
- `compare_models.py` - Script to compare and visualize model performances


## Installation

```bash
pip install torch torchvision matplotlib numpy pandas
```

## How to Run

### Step 1: Train Individual Models

Each model can be trained separately:

```bash
# Train AlexNet
python alexnet_cifar10.py

# Train VGG16
python vgg_cifar10.py

# Train VGG16 with Batch Normalization
python vgg_bn_cifar10.py

# Train Reduced VGG (VGG8)
python vgg_reduced_cifar10.py
```

Each script will:
- Load and preprocess the CIFAR-10 dataset
- Initialize the respective CNN model
- Train for 50 epochs
- Evaluate on the test set
- Save model weights, metrics, and visualizations

### Step 2: Compare Models

After training all models (or using the provided metrics files), run:

```bash
python compare_models.py
```

This will:
- Load metrics from each model
- Generate comparative visualizations in `figures/comparison/`
- Create a summary table as CSV

## Model Performance Summary

Here's a comparison of the models on the CIFAR-10 dataset:

| Model    | Parameters (M) | Test Accuracy (%) | Avg Time/Epoch (s) | Max Val Accuracy (%) | Final Val Loss |
|----------|----------------|-------------------|--------------------|---------------------|----------------|
| alexnet  | 35.86          | 89.28             | 22.09              | 89.44               | 0.4126         |
| vgg16    | 33.64          | 88.23             | 26.50              | 89.28               | 0.4106         |
| vgg16_bn | 33.65          | 91.62             | 27.84              | 91.83               | 0.3696         |
| vgg8     | 7.13           | 88.42             | 21.37              | 88.43               | 0.4254         |

## Key Findings

- VGG16-BN achieves the highest accuracy (91.62%) due to better regularization from batch normalization
- VGG8 is the most parameter-efficient model, achieving comparable accuracy to VGG16 with ~78% fewer parameters
- AlexNet shows impressive efficiency despite being an older architecture
- Batch normalization provides substantial performance improvements at minimal computational cost

## Visualizations

The project generates several visualizations:
- Feature maps for each model
- Training curves (accuracy and loss)
- Overfitting analysis
- Performance vs. model size comparisons
- Time per epoch comparisons
