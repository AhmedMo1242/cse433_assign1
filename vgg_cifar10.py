import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for figures
os.makedirs('figures/vgg', exist_ok=True)

# Data Preprocessing and Loading
def load_cifar10():
    """
    Loads and preprocesses the CIFAR-10 dataset.
    
    Applies data augmentation for training (random crop, flip) and 
    normalizes using CIFAR-10 mean and std values.
    
    Returns:
        tuple: Contains:
            - trainloader (DataLoader): DataLoader for training data
            - testloader (DataLoader): DataLoader for test data
            - classes (tuple): Class names of CIFAR-10
    """
    # CIFAR-10 mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# VGG Model Definition (Adapted for CIFAR-10)
class VGG16(nn.Module):
    """
    VGG16 architecture implementation for CIFAR-10.
    
    The network contains 13 convolutional layers and 3 fully connected layers
    arranged in blocks. Each block uses multiple 3x3 convolutions followed by
    max pooling to reduce spatial dimensions.
    
    Args:
        num_classes (int): Number of output classes. Default is 10 for CIFAR-10.
    """
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers - adjusted for CIFAR-10's dimensions
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass of the VGG16 model.
        
        Args:
            x (Tensor): Input images of shape [batch_size, 3, 32, 32]
            
        Returns:
            Tensor: Class logits of shape [batch_size, num_classes]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """
        Initialize the weights of convolutional and linear layers.
        
        Uses Kaiming initialization for convolutional layers and
        normal distribution for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Training and Evaluation Functions
def train_model(model, trainloader, epochs=50):
    """
    Train the model on the CIFAR-10 training data.
    
    Uses SGD optimizer with momentum and weight decay. Learning rate is
    adjusted based on validation loss using ReduceLROnPlateau scheduler.
    
    Args:
        model (nn.Module): The model to train
        trainloader (DataLoader): DataLoader for training data
        epochs (int): Number of training epochs
        
    Returns:
        tuple: Contains:
            - model (nn.Module): Trained model
            - history (dict): Training metrics history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'time_per_epoch': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
                running_loss = 0.0
        
        # Calculate epoch metrics
        train_loss, train_acc = evaluate_model(model, trainloader)
        val_loss, val_acc = evaluate_model(model, testloader)
        epoch_time = time.time() - start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['time_per_epoch'].append(epoch_time)
        
        print(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_loss)
    
    return model, history

def evaluate_model(model, dataloader):
    """
    Evaluate the model on the provided data.
    
    Computes loss and accuracy on the dataset without updating model parameters.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader containing evaluation data
        
    Returns:
        tuple: Contains:
            - loss (float): Average loss over the dataset
            - accuracy (float): Classification accuracy as percentage
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

# Visualization Functions
def visualize_filters(model, layer_idx=0, filename='vgg_filters.png'):
    """
    Visualize filters from a convolutional layer of the model.
    
    Args:
        model (nn.Module): The model containing filters to visualize
        layer_idx (int): Index of the convolutional layer to visualize
        filename (str): Output filename for the visualization
    """
    # Extract filters from the first convolutional layer
    filters = model.features[layer_idx].weight.data.cpu().numpy()
    
    # Normalize filter values between 0 and 1
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot filters
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            # For each filter
            img = filters[i, 0, :, :]  # First channel only
            ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'figures/vgg/{filename}')
    plt.close()

def visualize_feature_maps(model, dataloader, filename='vgg_feature_maps.png'):
    """
    Visualize feature maps activated by an input image.
    
    Uses forward hooks to capture activations from convolutional layers.
    
    Args:
        model (nn.Module): The model to extract feature maps from
        dataloader (DataLoader): DataLoader containing input images
        filename (str): Output filename for the visualization
    """
    # Get a batch of images
    dataiter = iter(dataloader)
    images, _ = next(dataiter)
    image = images[0:1].to(device)  # Take the first image
    
    # Create hooks to capture feature maps
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize feature maps from the first few conv layers
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    i = 0
    for name, feat_map in activations.items():
        if i >= len(axes):
            break
            
        # Get feature map from the first channel
        feat_map = feat_map[0, 0, :, :].cpu().numpy()
        axes[i].imshow(feat_map, cmap='viridis')
        axes[i].set_title(f'Layer: {name}')
        axes[i].axis('off')
        i += 1
    
    plt.tight_layout()
    plt.savefig(f'figures/vgg/{filename}')
    plt.close()

def plot_training_history(history, filename='vgg_training_history.png'):
    """
    Plot training and validation metrics history.
    
    Creates plots for loss, accuracy, and training time per epoch.
    
    Args:
        history (dict): Dictionary containing training history
        filename (str): Base filename for the output plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/vgg/{filename}')
    plt.close()
    
    # Also plot time per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(history['time_per_epoch'])
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Time per Epoch')
    plt.grid(True)
    plt.savefig(f'figures/vgg/vgg_time_per_epoch.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader, classes = load_cifar10()
    
    print("Creating VGG16 model...")
    vgg16 = VGG16(num_classes=10)
    total_params = sum(p.numel() for p in vgg16.parameters())
    print(f"VGG16 Total parameters: {total_params:,}")
    
    print("Training VGG16 model...")
    vgg16, history = train_model(vgg16, trainloader, epochs=50)
    
    print("Evaluating VGG16 on test data...")
    test_loss, test_acc = evaluate_model(vgg16, testloader)
    print(f"VGG16 Test Accuracy: {test_acc:.2f}%")
    print(f"VGG16 Test Loss: {test_loss:.4f}")
    
    # Print average time per epoch
    avg_time = sum(history['time_per_epoch']) / len(history['time_per_epoch'])
    print(f"VGG16 Average time per epoch: {avg_time:.2f} seconds")
    
    # Save model
    torch.save(vgg16.state_dict(), 'vgg16_cifar10.pth')
    
    # Visualizations
    print("Generating visualizations...")
    visualize_filters(vgg16)
    visualize_feature_maps(vgg16, testloader)
    plot_training_history(history)
    
    # Save metrics to JSON for comparison
    import json
    metrics = {
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc'],
        'time_per_epoch': history['time_per_epoch'],
        'test_acc': test_acc,
        'params': total_params
    }
    
    with open('vgg16_history.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to vgg16_history.json")
    
    print("VGG16 analysis completed!")
