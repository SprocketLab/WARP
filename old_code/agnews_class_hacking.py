import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import os
import json
import pickle
from datetime import datetime
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Class distribution estimation with BERT on AG News")
parser.add_argument('--num_interpolations', type=int, default=15, help='Number of interpolation steps (K)')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for fine-tuning')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for gradient computation')
parser.add_argument('--num_epochs', type=int, default=4, help='Number of fine-tuning epochs')
parser.add_argument('--n_total', type=int, default=5000, help='Total samples in dataset D')
parser.add_argument('--n_finetune', type=int, default=2500, help='Samples in fine-tuning subset D\'')
args = parser.parse_args()

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"class_distribution_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*70}\nOutput directory created: {output_dir}\n{'='*70}\n")

# Configuration
class Config:
    n_total = args.n_total
    n_finetune = args.n_finetune
    model_name = 'bert-base-uncased'
    max_length = 256
    num_labels = 4  # AG News has 4 classes
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    K = args.num_interpolations
    lambdas = np.linspace(0.05, 0.95, args.num_interpolations)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Load AG News dataset
print("="*70 + "\nSTEP 1: Loading AG News Dataset\n" + "="*70)
dataset = load_dataset('ag_news')
train_data = dataset['train']
print(f"Full AG News training set size: {len(train_data)}")

# Create subset D
indices_D = random.sample(range(len(train_data)), config.n_total)
D = train_data.select(indices_D)
print(f"Selected subset D with {config.n_total} samples")

# Create fine-tuning subset D' with skewed class distribution
class_weights = [0.4, 0.3, 0.2, 0.1]  # Skewed distribution: favor class 0, then 1, etc.
labels_all = np.array([train_data[i]['label'] for i in indices_D])
indices_D_prime = []
for class_id in range(config.num_labels):
    class_indices = np.where(labels_all == class_id)[0]
    n_samples = int(config.n_finetune * class_weights[class_id])
    selected = random.sample(list(class_indices), min(n_samples, len(class_indices)))
    indices_D_prime.extend(selected)
# Adjust to exactly n_finetune samples
if len(indices_D_prime) > config.n_finetune:
    indices_D_prime = random.sample(indices_D_prime, config.n_finetune)
elif len(indices_D_prime) < config.n_finetune:
    remaining = random.sample(list(set(range(config.n_total)) - set(indices_D_prime)), 
                             config.n_finetune - len(indices_D_prime))
    indices_D_prime.extend(remaining)
print(f"Selected fine-tuning subset D' with {config.n_finetune} samples (skewed distribution)")

# Compute ground-truth class distribution
labels_D_prime = [D[i]['label'] for i in indices_D_prime]
class_counts = np.bincount(labels_D_prime, minlength=config.num_labels)
p_prime = class_counts / config.n_finetune
print(f"\nDataset Statistics:\n  Total samples |D|: {config.n_total}\n  Fine-tuning samples |D'|: {config.n_finetune}")
print(f"  Ratio |D'|/|D|: {config.n_finetune/config.n_total:.2%}")
print(f"  Ground-truth class distribution p': {p_prime}")

# Save dataset info
dataset_info = {
    'n_total': config.n_total, 'n_finetune': config.n_finetune, 'indices_D': indices_D,
    'indices_D_prime': indices_D_prime, 'class_distribution': p_prime.tolist(),
    'class_weights': class_weights
}
with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)
print(f"✓ Dataset info saved to {output_dir}/dataset_info.json")

# Custom Dataset
class AGNewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and datasets
tokenizer = BertTokenizer.from_pretrained(config.model_name)
D_dataset = AGNewsDataset(D, tokenizer, config.max_length)
D_prime_dataset = Subset(D_dataset, indices_D_prime)
D_loader = DataLoader(D_dataset, batch_size=config.batch_size, shuffle=False)
D_prime_loader = DataLoader(D_prime_dataset, batch_size=config.batch_size, shuffle=True)

# Training function
def train_model(model, dataloader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        print(f"Epoch {epoch+1} - Average Loss: {total_loss / len(dataloader):.4f}")

# Parameter and gradient utilities
def get_last_layer_params(model):
    return {name: param for name, param in model.named_parameters() if 'classifier' in name}

def extract_last_layer_gradient(model):
    grad = [param.grad.flatten().clone() for name, param in model.named_parameters() if 'classifier' in name and param.grad is not None]
    return torch.cat(grad) if grad else None

# Initialize and save base model
print("\n" + "="*70 + "\nSTEP 2: Initializing Base Model (θ_base)\n" + "="*70)
theta_base_model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(config.device)
theta_base = {name: param.clone().detach() for name, param in theta_base_model.named_parameters()}
torch.save(theta_base_model.state_dict(), os.path.join(output_dir, 'theta_base_model.pt'))
print(f"Base model loaded: {config.model_name}\nNumber of parameters: {sum(p.numel() for p in theta_base_model.parameters()):,}")
print(f"✓ Base model saved to {output_dir}/theta_base_model.pt")

# Fine-tune expert model
print("\n" + "="*70 + "\nSTEP 3: Fine-tuning on D' to Create Expert Model (θ_exp)\n" + "="*70)
theta_exp_model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(config.device)
optimizer = AdamW(theta_exp_model.parameters(), lr=config.learning_rate)
print(f"Fine-tuning configuration:\n  Learning rate: {config.learning_rate}\n  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}\n  Training samples: {config.n_finetune}\n  Batches per epoch: {len(D_prime_loader)}")
train_model(theta_exp_model, D_prime_loader, optimizer, config.device, config.num_epochs)
theta_exp = {name: param.clone().detach() for name, param in theta_exp_model.named_parameters()}
torch.save(theta_exp_model.state_dict(), os.path.join(output_dir, 'theta_exp_model.pt'))
print(f"✓ Expert model saved to {output_dir}/theta_exp_model.pt")

# Last layer analysis
theta_base_last = get_last_layer_params(theta_base_model)
theta_exp_last = get_last_layer_params(theta_exp_model)
param_distance = np.sqrt(sum(torch.norm(theta_exp_last[name] - theta_base_last[name]).item() ** 2 for name in theta_base_last))
print(f"\nLast layer parameters:\n  Total: {sum(p.numel() for p in theta_base_last.values()):,}")
print(f"  Ratio to full model: {sum(p.numel() for p in theta_base_last.values()) / sum(p.numel() for p in theta_base_model.parameters()):.2%}")
print(f"Last layer parameter distance ||θ_exp - θ_base||: {param_distance:.4f}")

# Compute gradients and alignment matrices
print("\n" + "="*70 + "\nSTEP 4: Computing Alignment Matrices M and M_c\n" + "="*70)
print(f"Configuration:\n  Number of interpolated models (K): {config.K}\n  Lambda values: {config.lambdas}")
M = np.zeros((config.n_total, config.K))
M_c = np.zeros((config.num_labels, config.K))

for k, lambda_k in enumerate(config.lambdas):
    print(f"\n{'-'*70}\nInterpolated Model {k+1}/{config.K} (λ={lambda_k:.2f})\n{'-'*70}")
    theta_k_model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(config.device)
    with torch.no_grad():
        for name, param in theta_k_model.named_parameters():
            param.copy_((1 - lambda_k) * theta_base[name] + lambda_k * theta_exp[name])
    direction = torch.cat([theta_k_model.state_dict()[name] - theta_exp_last[name] for name in theta_base_last]).flatten()
    direction_norm = torch.norm(direction).item()
    print(f"Last layer direction norm ||θ_k_last - θ_exp_last||: {direction_norm:.4f}")
    
    # Compute gradients once and reuse
    theta_k_model.eval()
    sample_grads = []
    class_grad_sums = {c: None for c in range(config.num_labels)}
    class_counts = {c: 0 for c in range(config.num_labels)}
    sample_idx = 0
    
    for batch in tqdm(D_loader, desc="Computing gradients"):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        for i in range(input_ids.size(0)):
            theta_k_model.zero_grad()
            outputs = theta_k_model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1], labels=labels[i:i+1])
            loss = outputs.loss
            loss.backward()
            grad_i = extract_last_layer_gradient(theta_k_model)
            if grad_i is None:
                sample_grads.append(None)
                continue
            sample_grads.append(grad_i)
            class_id = labels[i].item()
            class_counts[class_id] += 1
            class_grad_sums[class_id] = grad_i.clone() if class_grad_sums[class_id] is None else class_grad_sums[class_id] + grad_i
            M[sample_idx, k] = torch.dot(grad_i, direction).item() / direction_norm
            sample_idx += 1
    
    # Compute class alignment scores
    for class_id in range(config.num_labels):
        if class_counts[class_id] > 0:
            grad = class_grad_sums[class_id] / class_counts[class_id]
            M_c[class_id, k] = torch.dot(grad, direction).item() / direction_norm
        else:
            M_c[class_id, k] = 0.0
    
    print(f"Class counts: {class_counts}\nClass alignment scores for λ={lambda_k:.2f}: {M_c[:, k]}")
    del theta_k_model
    torch.cuda.empty_cache()

np.save(os.path.join(output_dir, 'alignment_matrix_M.npy'), M)
np.save(os.path.join(output_dir, 'class_alignment_matrix_M_c.npy'), M_c)
print(f"\nAlignment Matrix M shape: {M.shape}\nStatistics:\n  Mean: {M.mean():.6f}\n  Std: {M.std():.6f}")
print(f"Class Alignment Matrix M_c shape: {M_c.shape}\nStatistics:\n  Mean: {M_c.mean():.6f}\n  Std: {M_c.std():.6f}")
print(f"✓ Alignment matrices saved to {output_dir}")

# Train regression model for class distribution
print("\n" + "="*70 + "\nSTEP 5: Training Regression Model to Predict Class Distribution p'\n" + "="*70)
regressor = LinearRegression()
cv_scores_mse = cross_val_score(regressor, M_c, p_prime, cv=3, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {-cv_scores_mse}\nMean CV MSE: {-cv_scores_mse.mean():.6f} (+/- {cv_scores_mse.std():.6f})")
regressor.fit(M_c, p_prime)
p_prime_pred = np.clip(regressor.predict(M_c), 0, 1)
p_prime_pred /= p_prime_pred.sum()
kl_div = np.sum(p_prime * np.log(p_prime / (p_prime_pred + 1e-10) + 1e-10))
print(f"KL-divergence: {kl_div:.6f}\nGround-truth class distribution p': {p_prime}\nPredicted class distribution p'_pred: {p_prime_pred}")
with open(os.path.join(output_dir, 'class_distribution_regressor.pkl'), 'wb') as f:
    pickle.dump(regressor, f)
print(f"✓ Regressor saved to {output_dir}/class_distribution_regressor.pkl")

# Save results
results = {
    'dataset': {
        'n_total': config.n_total,
        'n_finetune': config.n_finetune,
        'ratio': config.n_finetune / config.n_total,
        'class_distribution': p_prime.tolist(),
        'class_weights': class_weights
    },
    'model': {
        'base_model': config.model_name,
        'num_parameters': sum(p.numel() for p in theta_base_model.parameters()),
        'num_epochs': config.num_epochs,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size
    },
    'interpolation': {
        'K': config.K,
        'lambdas': config.lambdas.tolist()
    },
    'sample_alignment_matrix': {
        'shape': M.shape,
        'mean': float(M.mean()),
        'std': float(M.std()),
        'min': float(M.min()),
        'max': float(M.max())
    },
    'class_alignment_matrix': {
        'shape': M_c.shape,
        'mean': float(M_c.mean()),
        'std': float(M_c.std()),
        'min': float(M_c.min()),
        'max': float(M_c.max())
    },
    'class_distribution_estimation': {
        'cross_val_mse_scores': (-cv_scores_mse).tolist(),
        'cross_val_mse_mean': float(-cv_scores_mse.mean()),
        'cross_val_mse_std': float(cv_scores_mse.std()),
        'kl_divergence': float(kl_div),
        'predicted_distribution': p_prime_pred.tolist()
    }
}
with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved to {output_dir}/results.json")

predictions = {
    'class_distribution_true': p_prime.tolist(),
    'class_distribution_predicted': p_prime_pred.tolist()
}
with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
    json.dump(predictions, f, indent=2)
print(f"✓ Predictions saved to {output_dir}/predictions.json")

# Final summary
print("\n" + "="*70 + "\nEXPERIMENT COMPLETE - Summary\n" + "="*70)
print(f"\nOutput directory: {output_dir}\n\nSaved files:")
for i, file in enumerate(['dataset_info.json', 'theta_base_model.pt', 'theta_exp_model.pt', 
                         'alignment_matrix_M.npy', 'class_alignment_matrix_M_c.npy', 
                         'class_distribution_regressor.pkl', 'results.json', 'predictions.json'], 1):
    print(f"  {i}. {file}")
print(f"\nKey Results:\n  ✓ Class Distribution Estimation:\n    - KL-divergence: {kl_div:.6f}")
print(f"    - Mean CV MSE: {-cv_scores_mse.mean():.6f}\n    - Ground-truth p': {p_prime}")
print(f"    - Predicted p'_pred: {p_prime_pred}")
print("\n" + "="*70 + "\nClass Distribution Estimation Complete!\n" + "="*70)