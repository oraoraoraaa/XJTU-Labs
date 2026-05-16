import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import numpy as np
import argparse

# ----------------- Part 1: PyTorch CNN ----------------- #
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Standard CNN for 32x32 images (CIFAR-10)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def run_cnn(lr=0.001, epochs=5, subset_ratio=1.0):
    print(f"\n--- Running CNN with lr={lr}, epochs={epochs} ---")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    if subset_ratio < 1.0:
        train_size = int(len(trainset) * subset_ratio)
        trainset = torch.utils.data.Subset(trainset, range(train_size))
        test_size = int(len(testset) * subset_ratio)
        testset = torch.utils.data.Subset(testset, range(test_size))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()
    start_train_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")
    
    train_time = time.time() - start_train_time
    
    # Testing
    model.eval()
    all_preds = []
    all_labels = []
    
    start_test_time = time.time()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_time = time.time() - start_test_time

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"CNN Results (lr={lr}, epochs={epochs}):")
    print(f"Train Time: {train_time:.2f}s | Test Time: {test_time:.2f}s")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    return acc, prec, rec, f1, train_time, test_time

# ----------------- Part 2: Scikit-Learn Random Forest ----------------- #
def run_rf(n_estimators=100, max_depth=None, subset_ratio=1.0):
    print(f"\n--- Running Random Forest with n_estimators={n_estimators}, max_depth={max_depth} ---")
    
    # Random Forest works with 1D feature arrays rather than 3D images
    transform = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if subset_ratio < 1.0:
        train_size = int(len(trainset) * subset_ratio)
        trainset = torch.utils.data.Subset(trainset, range(train_size))
        test_size = int(len(testset) * subset_ratio)
        testset = torch.utils.data.Subset(testset, range(test_size))

    # Convert to NumPy for sklearn
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    start_prep = time.time()
    # Loading into memory
    X_train, y_train = next(iter(trainloader))
    X_test, y_test = next(iter(testloader))
    
    # Flatten images ([N, 3, 32, 32] -> [N, 3072])
    X_train = X_train.view(X_train.size(0), -1).numpy()
    X_test = X_test.view(X_test.size(0), -1).numpy()
    y_train = y_train.numpy()
    y_test = y_test.numpy()
    print(f"Data Prep Time: {time.time() - start_prep:.2f}s")

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    
    # Training
    start_train_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_train_time

    # Testing
    start_test_time = time.time()
    y_pred = rf.predict(X_test)
    test_time = time.time() - start_test_time

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"RF Results (trees={n_estimators}, max_depth={max_depth}):")
    print(f"Train Time: {train_time:.2f}s | Test Time: {test_time:.2f}s")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
    return acc, prec, rec, f1, train_time, test_time

if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=float, default=1.0, help="Ratio of data to use (e.g., 0.1 for 10%% to speed up tests)")
    args = parser.parse_args()

    # Create result directory
    os.makedirs('result', exist_ok=True)

    print("=== CIFAR-10 Classification Comparison ===")
    
    results = []

    # 1. Compare CNN parameters
    print("\n[Evaluating CNN Configurations]")
    for lr, epochs in [(0.001, 5), (0.001, 2), (0.01, 5)]:
        acc, prec, rec, f1, train_time, test_time = run_cnn(lr=lr, epochs=epochs, subset_ratio=args.subset)
        results.append({
            'Model': f'CNN (lr={lr}, ep={epochs})',
            'Algorithm': 'CNN',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Train Time(s)': train_time,
            'Test Time(s)': test_time
        })
    
    # 2. Compare RF parameters
    print("\n[Evaluating RF Configurations]")
    for n_est in [10, 50, 100]:
        acc, prec, rec, f1, train_time, test_time = run_rf(n_estimators=n_est, max_depth=None, subset_ratio=args.subset)
        results.append({
            'Model': f'RF (trees={n_est})',
            'Algorithm': 'RF',
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Train Time(s)': train_time,
            'Test Time(s)': test_time
        })
        
    # Create a DataFrame
    df = pd.DataFrame(results)
    
    # Print the table
    print("\n=== Final Results Table ===")
    print(df.to_string(index=False))
    
    # Save the table outputs
    df.to_csv('result/metrics_comparison.csv', index=False)
    df.to_markdown('result/metrics_comparison.md', index=False)
    
    # Visualization
    print("\nSaving visualizations to 'result' folder...")
    
    # 1. Accuracy & F1-Score Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df['Model']))
    width = 0.35
    
    ax.bar(x - width/2, df['Accuracy'], width, label='Accuracy', color='skyblue')
    ax.bar(x + width/2, df['F1-Score'], width, label='F1-Score', color='salmon')
    
    ax.set_ylabel('Scores')
    ax.set_title('Accuracy and F1-Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('result/accuracy_vs_f1.png', dpi=300)
    plt.close()
    
    # 2. Training Time vs Test Time Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, df['Train Time(s)'], width, label='Train Time', color='lightgreen')
    ax.bar(x + width/2, df['Test Time(s)'], width, label='Test Time', color='orange')
    
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training and Test Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('result/time_comparison.png', dpi=300)
    plt.close()

    print("Done! Visualizations and tables are saved in 'result/' directory.")
