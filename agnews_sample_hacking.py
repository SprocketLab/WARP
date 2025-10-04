import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import random
import os
import json
import pickle
from datetime import datetime

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
output_dir = f"sample_hacking_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*70}")
print(f"Output directory created: {output_dir}")
print(f"{'='*70}\n")

# Configuration
class Config:
    # Dataset parameters
    n_total = 5000  # Total samples in D
    n_finetune = 2500  # Samples in D' (fine-tuning subset)
    
    # Model parameters
    model_name = 'bert-base-uncased'
    max_length = 256
    num_labels = 4  # AG News has 4 classes
    
    # Training parameters
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 4
    
    # Interpolation parameters
    K = 15  # Number of interpolated models
    lambdas = np.linspace(0.05, 0.95, K)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Load AG News dataset
print("="*70)
print("STEP 1: Loading AG News Dataset")
print("="*70)
dataset = load_dataset('ag_news')
train_data = dataset['train']
print(f"Full AG News training set size: {len(train_data)}")

# Create subset D of size n_total
indices_D = random.sample(range(len(train_data)), config.n_total)
D = train_data.select(indices_D)
print(f"Selected subset D with {config.n_total} samples")

# Create fine-tuning subset D' of size n_finetune
indices_D_prime = random.sample(range(config.n_total), config.n_finetune)
D_prime_global_indices = [indices_D[i] for i in indices_D_prime]
print(f"Selected fine-tuning subset D' with {config.n_finetune} samples")

# Create indicator vector I
I = np.zeros(config.n_total, dtype=int)
I[indices_D_prime] = 1

print(f"\nDataset Statistics:")
print(f"  Total samples |D|: {config.n_total}")
print(f"  Fine-tuning samples |D'|: {config.n_finetune}")
print(f"  Ratio |D'|/|D|: {config.n_finetune/config.n_total:.2%}")
print(f"  Indicator vector sum: {I.sum()}")
print(f"  Positive class ratio: {I.mean():.2%}")

# Save dataset info
dataset_info = {
    'n_total': config.n_total,
    'n_finetune': config.n_finetune,
    'indices_D': indices_D,
    'indices_D_prime': indices_D_prime,
    'indicator_vector': I.tolist()
}
with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)
print(f"\n✓ Dataset info saved to {output_dir}/dataset_info.json")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(config.model_name)

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
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
D_dataset = AGNewsDataset(D, tokenizer, config.max_length)
D_prime_dataset = Subset(D_dataset, indices_D_prime)

# DataLoaders
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
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

# Initialize base model
print("\n" + "="*70)
print("STEP 2: Initializing Base Model (θ_base)")
print("="*70)
theta_base_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)

# Save base model parameters
theta_base = {name: param.clone().detach() for name, param in theta_base_model.named_parameters()}
print(f"Base model loaded: {config.model_name}")
print(f"Number of parameters: {sum(p.numel() for p in theta_base_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in theta_base_model.parameters() if p.requires_grad):,}")

# Save base model
torch.save(theta_base_model.state_dict(), os.path.join(output_dir, 'theta_base_model.pt'))
print(f"✓ Base model saved to {output_dir}/theta_base_model.pt")

# Fine-tune on D' to get expert model
print("\n" + "="*70)
print("STEP 3: Fine-tuning on D' to Create Expert Model (θ_exp)")
print("="*70)
theta_exp_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)

optimizer = AdamW(theta_exp_model.parameters(), lr=config.learning_rate)
print(f"Fine-tuning configuration:")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Training samples: {config.n_finetune}")
print(f"  Batches per epoch: {len(D_prime_loader)}")

train_model(theta_exp_model, D_prime_loader, optimizer, config.device, config.num_epochs)

# Save expert model parameters
theta_exp = {name: param.clone().detach() for name, param in theta_exp_model.named_parameters()}

# Save expert model
torch.save(theta_exp_model.state_dict(), os.path.join(output_dir, 'theta_exp_model.pt'))
print(f"✓ Expert model saved to {output_dir}/theta_exp_model.pt")

# Compute parameter distance
param_distance = 0
for name in theta_base.keys():
    param_distance += torch.norm(theta_exp[name] - theta_base[name]).item() ** 2
param_distance = np.sqrt(param_distance)
print(f"\nParameter space distance ||θ_exp - θ_base||: {param_distance:.4f}")

print("\n" + "="*70)
print("STEP 4: Computing Alignment Matrix M")
print("="*70)
print(f"Configuration:")
print(f"  Number of interpolated models (K): {config.K}")
print(f"  Lambda values: {config.lambdas}")
print(f"  Total gradient computations: {config.n_total * config.K:,}")

# Initialize alignment matrix M: N x K
M = np.zeros((config.n_total, config.K))

# Compute per-sample gradients and alignment scores for each interpolated model
for k, lambda_k in enumerate(config.lambdas):
    print(f"\n{'-'*70}")
    print(f"Interpolated Model {k+1}/{config.K} (λ={lambda_k:.2f})")
    print(f"{'-'*70}")
    
    # Create interpolated model: θ_k = (1 - λ_k) * θ_base + λ_k * θ_exp
    theta_k_model = BertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=config.num_labels
    ).to(config.device)
    
    with torch.no_grad():
        for name, param in theta_k_model.named_parameters():
            param.copy_((1 - lambda_k) * theta_base[name] + lambda_k * theta_exp[name])
    
    # Compute direction: θ_k - θ_exp (flattened)
    direction = []
    for name in theta_base.keys():
        diff = theta_k_model.state_dict()[name] - theta_exp[name]
        direction.append(diff.flatten())
    direction = torch.cat(direction)
    direction_norm = torch.norm(direction).item()
    print(f"Direction norm ||θ_k - θ_exp||: {direction_norm:.4f}")
    
    # Compute per-sample gradients and alignment scores
    theta_k_model.eval()
    
    sample_idx = 0
    batch_alignment_scores = []
    
    for batch in tqdm(D_loader, desc=f"Computing gradients"):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        batch_size_actual = input_ids.size(0)
        
        # Compute per-sample gradients
        for i in range(batch_size_actual):
            theta_k_model.zero_grad()
            
            outputs = theta_k_model(
                input_ids=input_ids[i:i+1],
                attention_mask=attention_mask[i:i+1],
                labels=labels[i:i+1]
            )
            loss = outputs.loss
            loss.backward()
            
            # Collect gradient g_i^k (flattened)
            grad_i = []
            for name, param in theta_k_model.named_parameters():
                if param.grad is not None:
                    grad_i.append(param.grad.flatten().clone())
            grad_i = torch.cat(grad_i)
            
            # Compute alignment score: m_i^k = <g_i^k, θ_k - θ_exp> / ||θ_k - θ_exp||
            alignment_score = torch.dot(grad_i, direction).item() / direction_norm
            M[sample_idx, k] = alignment_score
            batch_alignment_scores.append(alignment_score)
            
            sample_idx += 1
    
    # Print statistics for this interpolated model
    col_scores = M[:, k]
    print(f"Alignment scores for λ={lambda_k:.2f}:")
    print(f"  Mean: {col_scores.mean():.6f}")
    print(f"  Std:  {col_scores.std():.6f}")
    print(f"  Min:  {col_scores.min():.6f}")
    print(f"  Max:  {col_scores.max():.6f}")
    
    # Compare alignment scores for samples in D' vs not in D'
    scores_in_D_prime = col_scores[I == 1]
    scores_not_in_D_prime = col_scores[I == 0]
    print(f"  Mean (in D'):     {scores_in_D_prime.mean():.6f}")
    print(f"  Mean (not in D'): {scores_not_in_D_prime.mean():.6f}")
    print(f"  Difference:       {scores_in_D_prime.mean() - scores_not_in_D_prime.mean():.6f}")
    
    del theta_k_model
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("Alignment Matrix M Computation Complete!")
print("="*70)
print(f"M shape: {M.shape}")
print(f"M statistics:")
print(f"  Global Mean: {M.mean():.6f}")
print(f"  Global Std:  {M.std():.6f}")
print(f"  Global Min:  {M.min():.6f}")
print(f"  Global Max:  {M.max():.6f}")

# Save alignment matrix
np.save(os.path.join(output_dir, 'alignment_matrix_M.npy'), M)
print(f"✓ Alignment matrix M saved to {output_dir}/alignment_matrix_M.npy")

# Save detailed statistics per lambda
lambda_stats = []
for k, lambda_k in enumerate(config.lambdas):
    col_scores = M[:, k]
    scores_in = col_scores[I == 1]
    scores_out = col_scores[I == 0]
    lambda_stats.append({
        'lambda': float(lambda_k),
        'mean': float(col_scores.mean()),
        'std': float(col_scores.std()),
        'min': float(col_scores.min()),
        'max': float(col_scores.max()),
        'mean_in_D_prime': float(scores_in.mean()),
        'mean_not_in_D_prime': float(scores_out.mean()),
        'difference': float(scores_in.mean() - scores_out.mean())
    })

with open(os.path.join(output_dir, 'lambda_statistics.json'), 'w') as f:
    json.dump(lambda_stats, f, indent=2)
print(f"✓ Lambda statistics saved to {output_dir}/lambda_statistics.json")

# Train classifier to predict indicator vector I
print("\n" + "="*70)
print("STEP 5: Training Classifier to Predict Indicator Vector I")
print("="*70)

# Use logistic regression
clf = LogisticRegression(max_iter=1000, random_state=42)
print(f"Classifier: Logistic Regression")
print(f"Features: Alignment matrix M ({M.shape[0]} samples × {M.shape[1]} features)")
print(f"Target: Indicator vector I (binary classification)")

# Cross-validation
print(f"\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(clf, M, I, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full data
print(f"\nTraining on full dataset...")
clf.fit(M, I)
I_pred = clf.predict(M)
accuracy = (I_pred == I).mean()
print(f"Training accuracy: {accuracy:.4f}")

# Save classifier
with open(os.path.join(output_dir, 'classifier.pkl'), 'wb') as f:
    pickle.dump(clf, f)
print(f"✓ Classifier saved to {output_dir}/classifier.pkl")

# Compute additional metrics
print("\n" + "="*70)
print("STEP 6: Evaluation Metrics")
print("="*70)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

precision = precision_score(I, I_pred)
recall = recall_score(I, I_pred)
f1 = f1_score(I, I_pred)

# Get prediction probabilities for ROC-AUC
I_pred_proba = clf.predict_proba(M)[:, 1]
roc_auc = roc_auc_score(I, I_pred_proba)

print(f"\nClassification Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({(I_pred == I).sum()}/{len(I)} correct)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(I, I_pred)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Not in D'  In D'")
print(f"Actual Not in D'  {cm[0,0]:6d}    {cm[0,1]:6d}")
print(f"Actual In D'      {cm[1,0]:6d}    {cm[1,1]:6d}")
print(f"\nInterpretation:")
print(f"  True Negatives:  {cm[0,0]} (correctly identified as NOT in D')")
print(f"  False Positives: {cm[0,1]} (incorrectly identified as in D')")
print(f"  False Negatives: {cm[1,0]} (missed samples that ARE in D')")
print(f"  True Positives:  {cm[1,1]} (correctly identified as in D')")

# Classification report
print(f"\nDetailed Classification Report:")
print(classification_report(I, I_pred, target_names=['Not in D\'', 'In D\'']))

# Analyze feature importance
print("\n" + "="*70)
print("STEP 7: Feature Importance Analysis")
print("="*70)
print(f"\nLogistic Regression Coefficients (Feature Importance):")
print(f"{'Lambda':>10} {'Coefficient':>15} {'Abs(Coef)':>15}")
print(f"{'-'*40}")
for k, lambda_k in enumerate(config.lambdas):
    coef = clf.coef_[0][k]
    print(f"{lambda_k:>10.2f} {coef:>15.6f} {abs(coef):>15.6f}")

# Find most important lambda
most_important_idx = np.argmax(np.abs(clf.coef_[0]))
most_important_lambda = config.lambdas[most_important_idx]
most_important_coef = clf.coef_[0][most_important_idx]
print(f"\nMost important interpolation point:")
print(f"  λ = {most_important_lambda:.2f} with coefficient {most_important_coef:.6f}")

# Save all results
results = {
    'dataset': {
        'n_total': config.n_total,
        'n_finetune': config.n_finetune,
        'ratio': config.n_finetune / config.n_total
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
    'alignment_matrix': {
        'shape': M.shape,
        'mean': float(M.mean()),
        'std': float(M.std()),
        'min': float(M.min()),
        'max': float(M.max())
    },
    'classification': {
        'cross_val_scores': cv_scores.tolist(),
        'cross_val_mean': float(cv_scores.mean()),
        'cross_val_std': float(cv_scores.std()),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    },
    'confusion_matrix': {
        'true_negatives': int(cm[0,0]),
        'false_positives': int(cm[0,1]),
        'false_negatives': int(cm[1,0]),
        'true_positives': int(cm[1,1])
    },
    'feature_importance': {
        'coefficients': clf.coef_[0].tolist(),
        'intercept': float(clf.intercept_[0]),
        'most_important_lambda': float(most_important_lambda),
        'most_important_coefficient': float(most_important_coef)
    }
}

with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {output_dir}/results.json")

# Save predictions
predictions = {
    'indicator_vector_true': I.tolist(),
    'indicator_vector_predicted': I_pred.tolist(),
    'prediction_probabilities': I_pred_proba.tolist()
}
with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
    json.dump(predictions, f, indent=2)
print(f"✓ Predictions saved to {output_dir}/predictions.json")

print("\n" + "="*70)
print("EXPERIMENT COMPLETE - Summary")
print("="*70)
print(f"\nOutput directory: {output_dir}")
print(f"\nSaved files:")
print(f"  1. dataset_info.json - Dataset configuration and indices")
print(f"  2. theta_base_model.pt - Base model weights")
print(f"  3. theta_exp_model.pt - Expert model weights")
print(f"  4. alignment_matrix_M.npy - Alignment matrix (N×K)")
print(f"  5. lambda_statistics.json - Statistics per interpolation point")
print(f"  6. classifier.pkl - Trained logistic regression classifier")
print(f"  7. results.json - Comprehensive results and metrics")
print(f"  8. predictions.json - Predictions and probabilities")

print(f"\nKey Results:")
print(f"  ✓ Successfully reverse-engineered fine-tuning dataset")
print(f"  ✓ Classification accuracy: {accuracy:.2%}")
print(f"  ✓ ROC-AUC score: {roc_auc:.4f}")
print(f"  ✓ Correctly identified {cm[1,1]}/{config.n_finetune} samples in D'")
print(f"  ✓ False positive rate: {cm[0,1]}/{config.n_total - config.n_finetune} = {cm[0,1]/(config.n_total - config.n_finetune):.2%}")

print("\n" + "="*70)
print("Sample-Level Hacking Reverse Engineering Complete!")
print("="*70)