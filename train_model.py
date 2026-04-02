import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from model import ImprovedDeepfakeDetector, SimpleCNN

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    # Use relative path - adjust if your data is in a different location
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        # Fallback: try absolute path (for backward compatibility)
        data_dir = r"C:\Users\siddh\OneDrive\Desktop\deepfake\data"
        if not os.path.exists(data_dir):
            data_dir = "data"  # Final fallback
    batch_size = 32
    epochs = 5  # Increased epochs for better training
    learning_rate = 0.0001  # Lower learning rate for fine-tuning
    model_path = "cnn_model.pth"
    use_improved_model = True  # Set to True to use ResNet-based model

    # Data augmentation for training - helps prevent overfitting and improves generalization
    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet standard size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Validation/test transforms - no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("Loading datasets...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Training data not found at '{data_dir}'. "
            "Expected folder structure: data/fake and data/real"
        )
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    if len(full_dataset) < 10:
        raise ValueError("Dataset is too small. Add more images before training.")
    if len(full_dataset.classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, found: {full_dataset.classes}")

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create separate datasets with different transforms for validation/test
    # We need to recreate them with val_transform
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # Get indices from splits
    val_indices = val_ds.indices
    test_indices = test_ds.indices

    # Create subset datasets
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)
    test_ds = torch.utils.data.Subset(test_dataset, test_indices)

    # Set num_workers=0 on Windows to avoid multiprocessing issues
    import sys
    num_workers = 0 if sys.platform == 'win32' else 2
    pin_memory = torch.cuda.is_available()  # Only use pin_memory with GPU

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # Calculate class weights to handle imbalance
    class_counts = [0, 0]
    for _, label in train_ds:
        class_counts[label] += 1

    total = sum(class_counts)
    class_weights = [total / (2 * count) if count > 0 else 1.0 for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}")

    # Create model
    if use_improved_model:
        print("Using ImprovedDeepfakeDetector (ResNet18-based)...")
        model = ImprovedDeepfakeDetector(num_classes=2, pretrained=True).to(device)
    else:
        print("Using SimpleCNN...")
        model = SimpleCNN().to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop with validation
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_weights': class_weights,
                'classes': full_dataset.classes,
                'model_type': 'improved' if use_improved_model else 'simple',
                'img_size': 224 if use_improved_model else 128
            }, model_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        print()

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # Final evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    print("\nTest Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\nTrue Negatives (Real->Real): {cm[0][0]}")
    print(f"False Positives (Real->Fake): {cm[0][1]}")
    print(f"False Negatives (Fake->Real): {cm[1][0]}")
    print(f"True Positives (Fake->Fake): {cm[1][1]}")
