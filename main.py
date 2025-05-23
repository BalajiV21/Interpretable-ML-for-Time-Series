# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# PyTorch imports for model creation and training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Captum interpretability methods
from captum.attr import IntegratedGradients, DeepLift, GradientShap, KernelShap

import os
import random
import numpy as np
import torch

# Set seed value
SEED = 85

# Python built-in random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For reproducibility in DataLoader
g = torch.Generator()
g.manual_seed(SEED)

from scipy.io import arff


data, meta = arff.loadarff('BankNote-Authentication.arff')
df = pd.DataFrame(data)

# Clean the 'Class' column
# Convert class values from (1, 2) to (0, 1)
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')) - 1)

# Define features and target
X = df.drop('Class', axis=1).values
y = df['Class'].values

print(df['Class'].value_counts())

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# PCA for Exploratory Component Analysis (ECA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
plt.title("PCA - ECA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Fraud")
plt.grid(True)
plt.tight_layout()
plt.show()

# t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y)
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Fraud")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot correlation matrix
corr = pd.DataFrame(X_scaled).corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True, generator=g)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, generator=g)

# Define CNN model class
class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * input_dim, 2)  # Output is binary classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

# Define LSTM model class
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))


# Train models
trained_models = {}
metrics = {}
for name, ModelClass in zip(["CNN", "LSTM"], [SimpleCNN, SimpleLSTM]):
    print(f"\n Training {name}...")
    model = ModelClass(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    # Evaluate model
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    print(f" {name} - Accuracy: {acc:.2f}, F1: {f1:.2f}, AUC: {auc if not np.isnan(auc) else 'nan'}")
    trained_models[name] = model
    metrics[name] = {"acc": acc, "f1": f1, "auc": auc}


from captum.attr import IntegratedGradients, DeepLift, GradientShap, KernelShap
from sklearn.metrics import f1_score, roc_auc_score
import torch
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure test tensor requires gradients
X_test_tensor.requires_grad_()

methods = ["IntegratedGradients", "DeepLift", "GradientShap", "KernelShap"]
model_names = list(trained_models.keys())

results_auc = {m: [] for m in model_names}
results_f1 = {m: [] for m in model_names}
relevance_maps = {}

for model_name in model_names:
    model = trained_models[model_name]
    model.eval()
    relevance_maps[model_name] = []

    for method_name in methods:
        print(f"{model_name} + {method_name}")
        try:
            if method_name == "IntegratedGradients":
                method = IntegratedGradients(model)
                attr = method.attribute(X_test_tensor[:40], target=1, baselines=X_test_tensor[:40]*0)

            elif method_name == "DeepLift":
                method = DeepLift(model)
                attr = method.attribute(X_test_tensor[:50], target=1, baselines=X_test_tensor[:50]*0)

            elif method_name == "GradientShap":
                method = GradientShap(model)
                baseline_dist = torch.randn_like(X_test_tensor[:20]) * 0.001
                attr = method.attribute(X_test_tensor[:20], baselines=baseline_dist, target=1)

            elif method_name == "KernelShap":
                method = KernelShap(model)
                attr_list = []
                for i in range(10):  # Fewer samples due to KernelShap slowness
                    single_attr = method.attribute(
                        X_test_tensor[i].unsqueeze(0),
                        baselines=torch.zeros_like(X_test_tensor[i].unsqueeze(0)),
                        target=1
                    )
                    attr_list.append(single_attr.detach().numpy())
                attr = np.vstack(attr_list)

            else:
                continue

            # If attr is tensor, convert to numpy
            if isinstance(attr, torch.Tensor):
                relevance = attr.detach().numpy()
            else:
                relevance = attr

            # Use model predictions directly for performance metrics
            # Attribution-based prediction using relevance sum
            relevance_sum = np.sum(relevance, axis=1)
            
            # Use a dynamic threshold — median works well to split the data
            threshold = np.percentile(relevance_sum, 50)  # try 60, 70, or tune it
            
            # Predict class: 1 if relevance > threshold, else 0
            preds = (relevance_sum > threshold).astype(int)
            
            # Evaluate against true labels
            eval_y = y_test[:relevance.shape[0]]
            auc_val = roc_auc_score(eval_y, preds) if len(np.unique(eval_y)) > 1 else 0
            f1_val = f1_score(eval_y, preds, zero_division=0)



            results_auc[model_name].append(auc_val)
            results_f1[model_name].append(f1_val)
            relevance_maps[model_name].append(relevance)

        except Exception as e:
            print(f"⚠️ Error for {model_name} + {method_name}: {e}")
            results_auc[model_name].append(0)
            results_f1[model_name].append(0)
            relevance_maps[model_name].append(np.zeros((50, X_test_tensor.shape[1])))

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(methods))
width = 0.2

for metric_name, results in zip(["AUC", "F1-score"], [results_auc, results_f1]):
    plt.figure(figsize=(10, 5))
    for i, model in enumerate(model_names):
        values = results[model]
        bars = plt.bar(x + i * width, values, width=width, label=model)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height + 0.01,  # position slightly above the bar
                     f"{height:.2f}",
                     ha='center', va='bottom', fontsize=9)

    plt.xticks(x + width, methods)
    plt.xlabel("Interpretability Methods")
    plt.ylabel(metric_name)
    plt.title(f"Interpretability Comparison ({metric_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# TIC Index Heatmap: shows correlation among methods for each model
fig, axes = plt.subplots(1, len(model_names), figsize=(12, 4))
for i, model_name in enumerate(model_names):
    relevance_avg = [np.mean(r, axis=0) for r in relevance_maps[model_name]]
    tic_matrix = np.corrcoef(relevance_avg)
    sns.heatmap(tic_matrix, ax=axes[i], annot=True, cmap="viridis")
    axes[i].set_title(f"TIC Index - {model_name}")
    axes[i].set_xticklabels(methods, rotation=45)
    axes[i].set_yticklabels(methods, rotation=0)
plt.tight_layout()
plt.show()
