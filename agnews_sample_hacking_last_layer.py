# importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW,SGD
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
import tempfile
import sys
import yaml 
import shutil
from collections import Counter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add mergekit to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mergekit'))

try:
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions
    MERGEKIT_AVAILABLE = True
    print("✓ Mergekit imported successfully")
except ImportError as e:
    MERGEKIT_AVAILABLE = False
    print(f"⚠ Mergekit not available: {e}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

if len(sys.argv) < 2:
    print("Usage: python script.py <config.json>")
    sys.exit(1)
    
config_file_path = sys.argv[1]
print(f"Loading configuration from: {config_file_path}")

# Load JSON configuration
try:
    with open(config_file_path, 'r') as f:
        config_dict = json.load(f)
    print("✓ Configuration loaded successfully")
except FileNotFoundError:
    print(f"Error: Configuration file '{config_file_path}' not found")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in '{config_file_path}': {e}")
    sys.exit(1)

# Create Config class
class Config:
    def __init__(self, config_dict):
        # Assign all JSON parameters to class attributes
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Compute derived parameters
        self.lambdas = np.linspace(self.lambda_min, self.lambda_max, self.K)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create config object
config = Config(config_dict)


# Create output directory with timestamp
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_dir = f"sample_hacking_output_{timestamp}"
output_dir = config.experiment_name
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*70}")
print(f"Output directory created: {output_dir}")
print(f"{'='*70}\n")


# Load AG News dataset
print("="*70)
print(f"STEP 1: Loading the {config.dataset} Dataset")
print("="*70)
dataset = load_dataset(config.dataset)
train_data = dataset['train']
print(f"Full training set size: {len(train_data)}")

# Show all features/fields
print("Dataset features: " +  str(train_data.features))


# Filter out samples with label -1 (for SNLI dataset)
valid_indices = []
for idx in range(len(train_data)):
    if train_data[idx]['label'] >=0:
        valid_indices.append(idx)

print(f"Total samples in dataset: {len(train_data)}")
print(f"Valid samples (label != -1): {len(valid_indices)}")

# Check if we have enough valid samples
if len(valid_indices) < config.n_total:
    print(f"ERROR: Not enough valid samples!")
    print(f"  Requested: {config.n_total}")
    print(f"  Available: {len(valid_indices)}")
    sys.exit(1)

# Create subset D of size n_total
# D contains the indices wrt to the original dataset
# indices_D contain the indices wrt the original training dataset for the select seed dataset
# we do it randomly as we want the select seed dataset to represent the original training corpus well
indices_D = random.sample(valid_indices, config.n_total)
print(f"Selected subset D with {config.n_total} samples")
D = train_data.select(indices_D)



# get which indexes in D correspond to the particualr label 
# get the percentage of the label..multiply it with n_total , see the frequnecy and take that many first values 
# use random.sample to select the frequnecy indices from the particualt list
# if number needed is greater the the lenght fo the array , exit and display the error message 
# add those indices to the array

epsilon = 1e-6  # or 1e-9 for tighter tolerance

if abs(sum(config.proportionArr) - 1.0) > epsilon:
    print(f"Sum of proportions should be 1, got {sum(config.proportionArr)}")
    exit(1)
    
# indices of the finetuning dataset
indices_D_prime = []


# is there direct correspondence between label i and index i in proportion array ?? - yes!
# this would be a probem if we dont geenrae experients that cover from a partcualr min to a particualr max 



# here the fientunign set is wrt the select seed dataset  (D) , not within the valid indices
if(config.finetuning_source == "select"):
    # labels_set = set(data_labels)
    # print("labels set: " + str(labels_set))
    labels_indices = {label: [] for label in range(config.num_labels)}
    for idx in indices_D:
        labels_indices[train_data[idx]['label']].append(idx)

    # print("dictionary for label_indices: " + str(label_indices.keys()) )

    for label in range(config.num_labels):
        proportion = config.proportionArr[label]
        available = len(labels_indices[label])
        samples_needed = int(np.ceil(proportion * config.n_finetune))
        print(f"Label {label}....samples needed {samples_needed}.....needed proportion: {proportion}....actual proportion: {samples_needed/config.n_finetune}")
        if(samples_needed>available):
            print(f"Datapoints for class {label} are less. Pls lessen the proportion")
            exit(1)
        
        label_indices_label = labels_indices[label]
        random.shuffle(label_indices_label)
        indices_D_prime.extend(label_indices_label[:samples_needed])


# here the fientunign set is wrt the valid indices, not  wrt the select seed dataset (D)
elif (config.finetuning_source == "original"):
    labels_indices = {label: [] for label in range(config.num_labels)}
    for idx in valid_indices:
        labels_indices[train_data[idx]['label']].append(idx)
    
    for label_idx in range(config.num_labels):
        print("Label idx: " + str(label_idx))
        proportion_needed = config.proportionArr[label_idx]
        datapoints_needed = int(proportion_needed*config.n_finetune)
        print(datapoints_needed)
        if(datapoints_needed>len(labels_indices[label_idx])):
            print(f"Datapoints for class {label_idx} are less. Pls lessen the proportion")
            exit(1)
            
        label_indices = labels_indices[label_idx]
        random.shuffle(label_indices)
        label_indices = label_indices[:datapoints_needed]
        indices_D_prime.extend(label_indices)
        
    random.shuffle(indices_D_prime)
    
    
    
# we care that our final_arr is to the appropriate size
# we randomly remove the extra points. There will be always equal or extra points since we are using np.ceil
# also we assume the fientuning set and the proportion of each class is large enough to not affect it signficantly

print(f"Size of the computed finetuning set: {len(indices_D_prime)} ")
if(len(indices_D_prime)>config.n_finetune):
    # to cover the edge case of random choosing two points with the same value
    random_indices = random.sample(range(0,len(indices_D_prime)),len(indices_D_prime)-config.n_finetune)
    indices_D_prime = [indices_D_prime[i] for i in range(len(indices_D_prime)) if i not in random_indices]


print(f"Fine-tuning set expected size: {config.n_finetune}")
print(f"Size of the fixed/updated finetuning set: {len(indices_D_prime)} ")

# getting the fientuning dataset wrt the original training dataset
# D_prime_global_indices = [indices_D[i] for i in indices_D_prime]
# print(f"Selected fine-tuning subset D' with {config.n_finetune} samples")

print(f"\nDataset Statistics:")
print(f"  Total samples |D|: {config.n_total}")
print(f"  Fine-tuning samples |D'|: {config.n_finetune}")
print(f"  Ratio |D'|/|D|: {config.n_finetune/config.n_total:.2%}")

# Save dataset info
dataset_info = {
    'n_total': config.n_total,
    'n_finetune': config.n_finetune,
    'indices_D': indices_D,   # indices of select seed dataset
    'indices_D_prime': indices_D_prime,  # indices of fine-tuning dataset
}
with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)
print(f"\n✓ Dataset info saved to {output_dir}/dataset_info.json")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(config.model_name)

base_model_path = os.path.join(output_dir, 'base_model') 
expert_model_path = os.path.join(output_dir, 'expert_model') 

# Custom Dataset
class ExperimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # text = self.data[idx]['text']
        premise = self.data[idx]['premise']
        hypothesis = self.data[idx]['hypothesis']
        text = f"Premis: {premise} Hypothesis: {hypothesis}"
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

# Create datasets (for the subset and fine-tuning subset)
D_original = ExperimentDataset(train_data, tokenizer, config.max_length)
D_dataset = ExperimentDataset(D, tokenizer, config.max_length)
D_prime_dataset = ExperimentDataset(train_data.select(indices_D_prime), tokenizer,config.max_length)

# DataLoaders
D_loader = DataLoader(D_dataset, batch_size=config.batch_size, shuffle=False)
D_prime_loader = DataLoader(D_prime_dataset, batch_size=config.batch_size, shuffle=False )
# shuffling the fientunign set since random.sample returns the indexes in the sorted format..
# and the dataset itself might not be shuffled..so thats why shufflfing those points. The points
# remain the same but their distributiona cross any factor eg class is much more uniform. 


# the eval set is the same for any caller calling the eval function
def eval(model,device):
    model.eval()
    correct_pred = 0.0
    
    with torch.no_grad():
        # Get 5000 random indices
        no_of_eval_datapoints = 5000
        total_samples = len(train_data)
        random_indices = random.sample(range(total_samples), no_of_eval_datapoints)
        for i in random_indices:
            data = D_original[i]
            # unsqueeze is more compatible with CUDA
            input_ids = input_ids = data['input_ids'].to(device).unsqueeze(0)  
            attention_mask = data['attention_mask'].to(device).unsqueeze(0)
            label = data['labels'].to(device)
            outputs = model(input_ids,attention_mask)
            predictions = torch.argmax(torch.softmax(outputs.logits, dim=-1))
            # Compare prediction with true label
            if predictions.item() == label.item():
                correct_pred += 1
    return correct_pred/no_of_eval_datapoints
    
    
# Training function
def train_model(model, dataloader, optimizer, device, num_epochs):
    accuracy_arr = []
    model.train()
    batch_interval = round((num_epochs*len(dataloader))/((config.K + 1)))
    print("Batch Interval: " + str(batch_interval))
    num_batch = 0
    num_model = 0
    
    # saving the base model 
    # torch.save(model, os.path.join(output_dir, f'model_{num_model}.pt'))
    # accuracy_arr.append(eval(model,device))
    
    for epoch in range(num_epochs):
        total_loss = 0
        # tqdm is compatible with any iterable
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(labels.shape)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
            num_batch += 1
            
            if(num_batch%batch_interval==0) and num_model<config.K:
                print("current model number: " + str(num_model))
                print("current batch: " + str(num_batch))
                torch.save(model, os.path.join(output_dir, f'model_{num_model}.pt'))
                eval_accuracy = eval(model,device)
                print(f"Eval accuracy: {eval_accuracy}")
                accuracy_arr.append(eval_accuracy)
                model.train()
                num_model+=1
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    return accuracy_arr
        
    # saving the expert model
    # torch.save(model, os.path.join(output_dir, f'model_{num_model}.pt'))
    
def quadratic_interpolation_weight(lambda_val, curve_param=0.3):
    """
    Convert lambda to quadratic interpolation weight
    
    Args:
        lambda_val: Original lambda [0,1]
        curve_param: Curvature (0=linear, >0=convex, <0=concave)
    
    Returns:
        Quadratic weight for interpolation
    """
    # Quadratic function: w(λ) = aλ² + bλ + c
    # Constraints: w(0)=0, w(1)=1
    # This gives: w(λ) = curve_param*λ² + (1-curve_param)*λ
    return curve_param * lambda_val**2 + (1 - curve_param) * lambda_val

# λ (lambda) is the global scale on whatever “important changes” the method kept.


def create_slerp_config(base_model_path: str, expert_model_path: str, t: float, output_path: str) -> str:
    """
    Create YAML config for SLERP merge
    
    Args:
        base_model_path: Path to base model
        expert_model_path: Path to expert model
        t: Interpolation parameter [0, 1]
        output_path: Where to save merged model
        
    Returns:
        Path to YAML config file
    """
    
    t = float(t) 
    config = {
        'merge_method': 'slerp',
        'base_model': base_model_path,
        'slices': [
            {
                'sources': [
                    {'model': base_model_path, 'layer_range': [0, -1]},
                    {'model': expert_model_path, 'layer_range': [0, -1]}
                ]
            }
        ],
        'parameters': {
            't': t
        },
        'dtype': 'float32'
    }
    
    # Save to temp file
    config_path = os.path.join(tempfile.gettempdir(), f'slerp_config_{t:.3f}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def create_ties_config(base_model_path: str, expert_model_path: str, weight: float, 
                       density: float, output_path: str) -> str:
    """
    Create YAML config for TIES merge
    
    Args:
        base_model_path: Path to base model
        expert_model_path: Path to expert model
        weight: Weight for task vector (your λ)
        density: Fraction of parameters to keep
        output_path: Where to save merged model
        
    Returns:
        Path to YAML config file
    """
    
    weight = float(weight)  # ← ADD THIS LINE
    density = float(density)  # ← ADD THIS LINE
    
    config = {
        'merge_method': 'ties',
        'base_model': base_model_path,
        'models': [
            {
                'model': expert_model_path,
                'parameters': {
                    'weight': weight,
                    'density': density
                }
            }
        ],
        'parameters': {
            'normalize': False,
            'int8_mask': False
        },
        'dtype': 'float32'
    }
    
    config_path = os.path.join(tempfile.gettempdir(), f'ties_config_{weight:.3f}_{density:.2f}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def create_della_config(base_model_path: str, expert_model_path: str, weight: float,
                        density: float, epsilon: float, rescale: bool, output_path: str) -> str:
    """
    Create YAML config for DELLA merge
    
    Args:
        base_model_path: Path to base model
        expert_model_path: Path to expert model
        weight: Weight for task vector (your λ)
        density: Fraction of parameters to keep
        epsilon: Numerical stability constant
        rescale: Whether to rescale after pruning
        output_path: Where to save merged model
        
    Returns:
        Path to YAML config file
    """
    
    weight = float(weight)  # ← ADD THIS LINE
    density = float(density)  # ← ADD THIS LINE
    epsilon = float(epsilon)  # ← ADD THIS LINE
    
    config = {
        'merge_method': 'della',
        'base_model': base_model_path,
        'models': [
            {
                'model': expert_model_path,
                'parameters': {
                    'weight': weight,
                    'density': density,
                    'epsilon': epsilon
                }
            }
        ],
        'parameters': {
            'rescale': rescale
        },
        'dtype': 'float32'
    }
    
    config_path = os.path.join(tempfile.gettempdir(), f'della_config_{weight:.3f}_{density:.2f}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def merge_with_mergekit(
    base_model_path: str,
    expert_model_path: str,
    method: str,
    lambda_k: float,
    output_dir: str,
    **kwargs):
    """
    Merge models using mergekit's actual implementation
    
    Args:
        base_model_path: Path to base model (can be HF model or local path)
        expert_model_path: Path to expert model
        method: 'slerp', 'ties', or 'della'
        lambda_k: Interpolation parameter [0, 1]
        output_dir: Temporary directory for merged model
        **kwargs: Method-specific parameters
        
    Returns:
        Merged state dict
    """

    print(f"\n{'='*70}")
    print(f"Creating Pseudo-Expert with Mergekit {method.upper()}")
    print(f"λ = {lambda_k:.3f}")
    print(f"{'='*70}")
    
    if not MERGEKIT_AVAILABLE:
        raise ImportError("Mergekit not available. Cannot perform merge.")
    
    # Create temp output directory with unique ID to avoid conflicts
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    temp_output = os.path.join(tempfile.gettempdir(), f'merge_{method}_{lambda_k:.3f}_{unique_id}')
    os.makedirs(temp_output, exist_ok=True)
    print("made the directory")
    
    config_path = None
    
    try:
        # Create appropriate config
        if method == 'slerp':
            config_path = create_slerp_config(
                base_model_path, expert_model_path, lambda_k, temp_output
            )
        
        elif method == 'ties':
            density = kwargs.get('density', 0.9)
            config_path = create_ties_config(
                base_model_path, expert_model_path, lambda_k, density, temp_output
            )
        
        elif method == 'della':
            density = kwargs.get('density', 0.9)
            epsilon = kwargs.get('epsilon', 1e-8)
            rescale = kwargs.get('rescale', True)
            config_path = create_della_config(
                base_model_path, expert_model_path, lambda_k, 
                density, epsilon, rescale, temp_output
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  Config created: {config_path}")
        
        # Load YAML as dictionary FIRST
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        print(f"  Config dict loaded, keys: {list(config_dict.keys())}")
        
        # Try multiple methods to create MergeConfiguration
        merge_config = None
        last_error = None
        
        # Method 1: Direct instantiation with **kwargs
        merge_config = MergeConfiguration(**config_dict)
        print(f"  ✓ MergeConfiguration created via direct instantiation")
        
        if merge_config is None:
            raise Exception(f"Failed to create MergeConfiguration. Last error: {last_error}")
        
        # Create merge options
        merge_options = MergeOptions(
            copy_tokenizer=False,
            lazy_unpickle=False,
            low_cpu_memory=False,
            write_model_card=False
        )
        print(f"  ✓ MergeOptions created")


        # Run merge
        print(f"  Running mergekit {method.upper()} merge...")

        if merge_options:
            run_merge(merge_config, temp_output, merge_options)
        else:
            run_merge(merge_config, temp_output)
        
        print(f"  ✓ Merge completed")
        
        # Load merged model state dict
        print(f"  Loading merged model from: {temp_output}")
        merged_model = BertForSequenceClassification.from_pretrained(temp_output,num_labels=config.num_labels).to(config.device)
        
        return merged_model 
    
    
    # except ValueError as e: 
    #     if "Circular reference detected" in str(e):
    #         print(f"  ⚠ Ignoring Pydantic serialization error")
            
    #     # merged_model = BertForSequenceClassification.from_pretrained(temp_output,num_labels=config.num_labels).to(config.device)
    #     # return merged_model 
            
    except Exception as e:
        print(f"  ✗ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup temp directory
        if os.path.exists(temp_output):
            try:
                shutil.rmtree(temp_output)
                print(f"  ✓ Cleaned up temp directory")
            except Exception as e:
                print(f"  ⚠ Could not clean up temp directory: {e}")
                
            

def get_last_layer_params(model):
    """Extract only the classifier (last layer) parameters"""
    # ToDo: check if all lastlayer params have "classifier" in name
    # For BERT, the classifier is model.classifier
    last_layer_params = {}
    for name, param in model.named_parameters():
        if 'classifier' in name:  # Only get classifier layer
            last_layer_params[name] = param
    return last_layer_params

def extract_last_layer_gradient(model):
    grad = []
    for name, param in model.named_parameters():
        if 'classifier' in name and param.grad is not None:
            grad.append(param.grad.flatten().clone())
    if len(grad) == 0:
        return None
    return torch.cat(grad)

def get_interpolated_model(lambda_k,interpolation_name):
    print(f"\n{'-'*70}")
    print(f"Interpolated Model {k+1}/{config.K} (λ={lambda_k:.2f})")
    print(f"{'-'*70}")
    
    # Create interpolated model: θ_k = (1 - λ_k) * θ_base + λ_k * θ_exp
    # INTERPOLATE ALL PARAMETERS
    theta_k_model = BertForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=config.num_labels
    ).to(config.device)
    
    # ToDo: we need to chnage only the classification layer/last layer weights. 
    # What does the .named_parameters do and is there a better way to get the last layer weigths
    
    # Ans: we are not getting the last layer weights. By solving the prop equation, its importnant that we 
    # itnerpoalte all the parameters. 
    
    
    if(interpolation_name=='linear'):
        with torch.no_grad():
            for name, param in theta_k_model.named_parameters():
                param.copy_((1 - lambda_k) * theta_base[name] + lambda_k * theta_exp[name])
        return theta_k_model
    
    elif(interpolation_name=='quadratic'):
        quad_interpolation = quadratic_interpolation_weight(lambda_k, curve_param=0.3)
        with torch.no_grad():
            for name, param in theta_k_model.named_parameters():
                param.copy_((1 - quad_interpolation) * theta_base[name] + quad_interpolation * theta_exp[name])
        return theta_k_model
    
    else: 
        return merge_with_mergekit(base_model_path,expert_model_path,interpolation_name,lambda_k,tempfile.gettempdir())




"""
Base model intialising and directory creation
"""
# Initialize base model
print("\n" + "="*70)
print("STEP 2: Initializing Base Model (θ_base)")
print("="*70)
theta_base_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)

# Save base model parameters
theta_base = {name: param.clone().detach().to(config.device)  for name, param in theta_base_model.named_parameters()}
print(f"Base model loaded: {config.model_name}")
print(f"Number of parameters: {sum(p.numel() for p in theta_base_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in theta_base_model.parameters() if p.requires_grad):,}")

# NEW: Save as complete model directory (for mergekit)
base_model_dir = os.path.join(output_dir, 'base_model')
theta_base_model.save_pretrained(base_model_dir)
tokenizer.save_pretrained(base_model_dir)
print(f"✓ Base model saved to {base_model_dir}/ (for mergekit)")

# Keep .pt for backward compatibility
torch.save(theta_base_model.state_dict(), os.path.join(output_dir, 'theta_base_model.pt'))
print(f"✓ Base state dict saved to {output_dir}/theta_base_model.pt")

# Save base model
# torch.save(theta_base_model.state_dict(), os.path.join(output_dir, 'theta_base_model.pt'))
# print(f"✓ Base model saved to {output_dir}/theta_base_model.pt")




"""
Finetuning the base model
"""
# Fine-tune on D' to get expert model
print("\n" + "="*70)
print("STEP 3: Fine-tuning on D' to Create Expert Model (θ_exp)")
print("="*70)
theta_exp_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)

if(config.optimizer=="Adam"):
    # for adaptive learning rates
    optimizer = AdamW(theta_exp_model.parameters(), lr=config.learning_rate)
else:
    # for static learning rate
    optimizer = SGD(theta_exp_model.parameters(), lr=config.learning_rate)
    
print(f"Fine-tuning configuration:")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Training samples: {config.n_finetune}")
print(f"  Batches per epoch: {len(D_prime_loader)}")

accuracy_arr = train_model(theta_exp_model, D_prime_loader, optimizer, config.device, config.num_epochs)

with open(os.path.join(output_dir, 'accuracy_arr.pkl'), 'wb') as f:
    pickle.dump(accuracy_arr, f)





"""
getting the last layer params and seeing the difference between the base model and the expert model
"""
# Save expert model
# torch.save(theta_exp_model.state_dict(), os.path.join(output_dir, 'theta_exp_model.pt'))
# print(f"✓ Expert model saved to {output_dir}/theta_exp_model.pt")

# After saving theta_exp, modify the parameter storage:
# Store only last layer parameters (stores the params where .required_grad is False)
theta_base_last = get_last_layer_params(theta_base_model)
theta_exp_last = get_last_layer_params(theta_exp_model)

print(f"\nLast layer parameters:")
for name in theta_base_last.keys():
    print(f"  {name}: {theta_base_last[name].numel()} parameters")
total_last_layer = sum(p.numel() for p in theta_base_last.values())
print(f"  Total last layer parameters: {total_last_layer:,}")
print(f"  Ratio to full model: {total_last_layer / sum(p.numel() for p in theta_base_model.parameters()):.4%}")


# ToDo: Whats exactly the use for param distance ?  
# Ans: to see if fientuning the model has had an impact on the parameters

# Compute last layer parameter distance
param_distance = 0
for name in theta_base_last.keys():
    param_distance += torch.norm(theta_exp_last[name] - theta_base_last[name]).item() ** 2
param_distance = np.sqrt(param_distance)
print(f"\nLast layer parameter distance ||θ_exp - θ_base||: {param_distance:.4f}")





"""
Expert model intialising and directory creation
"""
# Save expert model parameters (for linear/quadratic interpolation)
theta_exp = {name: param.clone().detach().to(config.device) for name, param in theta_exp_model.named_parameters()}

# NEW: Save as complete model directory (for mergekit)
expert_model_dir = os.path.join(output_dir, 'expert_model')
theta_exp_model.save_pretrained(expert_model_dir)
tokenizer.save_pretrained(expert_model_dir)
print(f"✓ Expert model saved to {expert_model_dir}/ (for mergekit)")

# Keep .pt for backward compatibility
torch.save(theta_exp_model.state_dict(), os.path.join(output_dir, 'theta_exp_model.pt'))
print(f"✓ Expert state dict saved to {output_dir}/theta_exp_model.pt")



for interpolation_name in config.interpolations:
    print("\n" + "="*70)
    print("STEP 4: Computing Alignment Matrix M")
    print("="*70)
    print(f"Configuration:")
    print(f"Interpolation: {interpolation_name}")
    print(f"  Number of interpolated models (K): {config.K}")
    print(f"  Lambda values: {config.lambdas}")
    print(f"  Total gradient computations: {config.n_total * config.K:,}")

    # Initialize alignment matrix M: N x K (the no of datapoints in the seed select dattaset * the no of pseudoexperts)
    M = np.zeros((config.n_total, config.K))

    # Interpolate the parameters, Compute per-sample gradients and alignment scores for each interpolated model
    for k, lambda_k in enumerate(config.lambdas):

        if(interpolation_name!='model_baseline'):
            theta_k_model = get_interpolated_model(lambda_k,interpolation_name)
        else:
            theta_k_model = torch.load(f'./{output_dir}/model_{k}.pt',weights_only=False)
        
        # Compute direction using ONLY LAST LAYER: θ_k_last - θ_exp_last (flattened)
        direction = []
        for name in theta_base_last.keys():
            diff = theta_k_model.state_dict()[name] - theta_exp_last[name]
            direction.append(diff.flatten())
        direction = torch.cat(direction).to(config.device)
        direction_norm = torch.norm(direction).item()
        print(f"Last layer direction norm ||θ_k_last - θ_exp_last||: {direction_norm:.4f}")
        
        # Compute per-sample gradients and alignment scores
        
        # putting model in eval mode. gradients can still be computed in eval mode
        theta_k_model.eval()
        
        sample_idx = 0
        
        alignment_scores = []
        for batch in tqdm(D_loader, desc=f"Computing sample gradients"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # print("num_labels:", theta_k_model.num_labels)
            # print("labels dtype:", labels.dtype, "device:", labels.device, "shape:", labels.shape)
            # print("labels min/max:", labels.min().item(), labels.max().item())
            # print("unique labels (sample):", labels.unique()[:20])
            
            batch_size_actual = input_ids.size(0)
            
            # Compute per-sample gradients
            for i in range(batch_size_actual):
                
                # setting stored graidents to zero
                theta_k_model.zero_grad()
                
                outputs = theta_k_model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    labels=labels[i:i+1]
                )
                loss = outputs.loss
                
                # computing gradients
                loss.backward()
                
                # Extract ONLY LAST LAYER gradient g_i^k
                grad_i = extract_last_layer_gradient(theta_k_model)
                
                if grad_i is None:
                    # M[sample_idx, k] = 0.0
                    alignment_scores.append(0)
                    sample_idx += 1
                    continue
                
                # Compute sample-level alignment score using last layer gradient
                # Alignment: <grad_last_layer, direction_last_layer>
                alignment_score = torch.dot(grad_i, direction).item() / direction_norm
                # M[sample_idx, k] = alignment_score
                alignment_scores.append(alignment_score)
                sample_idx += 1
                
                
        # print(f"Raw alignment scores: {alignment_scores}")
        # applying softmax per pseudoexpert 
        # alignment_scores = torch.softmax(torch.tensor(alignment_scores),dim=0)
        alignment_scores = torch.tensor(alignment_scores, dtype=torch.float32)
        alignment_scores = alignment_scores / alignment_scores.max() 
        # print(f"Softmax alignment scores: {alignment_scores}")
        
        for i in range(sample_idx):
            M[i, k] = alignment_scores[i]
        
        # Print statistics for this interpolated model
        col_scores = M[:, k]
        
        if(interpolation_name!="model_baseline"):
            print(f"Alignment scores for λ={lambda_k:.2f}:")
        else:
            print(f"Alignment scores for model {k}:")
        
        print(f"  Mean: {col_scores.mean():.6f}")
        print(f"  Std:  {col_scores.std():.6f}")
        print(f"  Min:  {col_scores.min():.6f}")
        print(f"  Max:  {col_scores.max():.6f}")
        
        # deleting the current model and emptying cache
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


    alignment_matrix_dir = f"{config.dataset}_{interpolation_name}_{config.proportionArr}"
    # Save alignment matrix
    os.makedirs(alignment_matrix_dir, exist_ok=True)
    np.save(os.path.join(alignment_matrix_dir , f'alignment_matrix_{interpolation_name}.npy'), M)
    print(f"✓ Alignment matrix M saved to {alignment_matrix_dir}/alignment_matrix_{interpolation_name}.npy")

    # Save detailed statistics per lambda
    lambda_stats = []
    for k, lambda_k in enumerate(config.lambdas):
        col_scores = M[:, k]
        lambda_stats.append({
            'lambda': float(lambda_k),
            'mean': float(col_scores.mean()),
            'std': float(col_scores.std()),
            'min': float(col_scores.min()),
            'max': float(col_scores.max())
        })

    with open(os.path.join(alignment_matrix_dir, 'lambda_statistics.json'), 'w') as f:
        json.dump(lambda_stats, f, indent=2)
    print(f"✓ Lambda statistics saved to {alignment_matrix_dir}/lambda_statistics.json")




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
    }
}

with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {output_dir}/results.json")

for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path) and ".pt" in filename and "model_" in filename:
        os.remove(file_path)

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

print("\n" + "="*70)
print("Sample-Level Hacking Reverse Engineering Complete!")
print("="*70) 
