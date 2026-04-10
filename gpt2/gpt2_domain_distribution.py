"""
GPT-2 Domain Distribution Alignment Experiment

This script orchestrates a complete experiment for analyzing how different samples
align with model fine-tuning trajectories using GPT-2 models. The experiment consists
of several key steps:

1. **Configuration Loading**: Loads experiment parameters from a JSON config file
2. **Dataset Preparation**: Creates seed dataset D and fine-tuning dataset D'
3. **Model Fine-tuning**: Trains GPT-2 from base to expert, saving intermediate checkpoints
4. **Alignment Computation**: Computes alignment scores for all samples across pseudo-experts
5. **Results Saving**: Saves alignment matrices and statistics for analysis

The main workflow:
- Load configuration from JSON (dataset, model params, class distribution)
- Create seed dataset D (for alignment) and fine-tuning dataset D' (biased distribution)
- Fine-tune GPT-2 base model → expert model, save K intermediate pseudo-experts
- For each pseudo-expert, compute alignment scores for all samples in D
- Save alignment matrix M (N×K) and per-lambda statistics

Usage:
    python gpt2_domain_distribution.py <config.json>
    
"""


import os 
import sys

current_file = os.path.abspath(__file__)
bert_dir = os.path.dirname(current_file)
project_root = os.path.dirname(bert_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import torch
import torch.nn as nn
import numpy as np
from data import Dataset
from gpt2_alignment import Alignment
from gpt2_finetuning import Finetuning
from gpt2_models import Model
import json
from transformers import GPT2ForSequenceClassification,GPT2Config,GPT2Tokenizer
from datasets import load_dataset
import pickle as pl


"""
Load the JSON configuration
"""

if len(sys.argv) < 2:
    print("Usage: python script.py <config.json>")
    sys.exit(1)
    
config_file_path = sys.argv[1]
print(f"Loading configuration from: {config_file_path}")

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
    
class Config:
    """
    Configuration container for experiment parameters.
    
    This class dynamically loads all parameters from a JSON configuration file
    and computes derived parameters needed for the experiment.
    
    Dynamic Attributes (from JSON):
        n_total (int): Size of seed dataset for alignment computation
        n_finetune (int): Size of fine-tuning dataset
        model_name (str): HuggingFace model identifier (e.g., 'openai-community/gpt2')
        max_length (int): Maximum sequence length for tokenization
        num_labels (int): Number of classification labels
        batch_size (int): Training batch size
        learning_rate (float): Optimizer learning rate
        num_epochs (int): Number of training epochs
        K (int): Number of pseudo-expert interpolation points
        lambda_min (float): Minimum interpolation coefficient
        lambda_max (float): Maximum interpolation coefficient
        interpolations (list): List of interpolation methods to use
        optimizer (str): Optimizer name ('Adam' or 'SGD')
        dataset (str): Dataset name (e.g., 'ag_news', 'snli')
        proportionArr (list): Target class distribution for fine-tuning dataset
        finetuning_source (str): Source for fine-tuning samples ('select' or 'original')
        experiment_name (str): Name/path for experiment outputs
    
    Computed Attributes:
        lambdas (np.ndarray): Array of K interpolation coefficients from lambda_min to lambda_max
        device (torch.device): Computation device (CUDA if available, else CPU)
    """
    def __init__(self, config_dict):
        # Assign all JSON parameters to class attributes
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Compute derived parameters
        self.lambdas = np.linspace(self.lambda_min, self.lambda_max, self.K)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create config object
config = Config(config_dict)



"""
Extract parameters from config object for easy access
"""
batch_size = config.batch_size 
num_labels = config.num_labels
max_length = config.max_length
epochs = config.num_epochs
finetuning_source = config.finetuning_source
learning_rate = config.learning_rate
device = config.device
optimizer_name = config.optimizer
no_of_pseudoexperts = config.K
model_name = config.model_name
output_dir = config.experiment_name
num_labels = config.num_labels
dataset_name = config.dataset
n_seed = config.n_total 
n_finetune = config.n_finetune 
proportionArr = config.proportionArr



"""
Initialize the GPT-2 tokenizer and model save paths
"""
base_model_path = os.path.join(output_dir, 'base_model') 
expert_model_path = os.path.join(output_dir, 'expert_model') 

# Load GPT-2 tokenizer (all models share the same tokenizer)
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="openai-community/gpt2")

# Configure tokenizer for GPT-2 classification
tokenizer.padding_side = "left"  # Left padding is standard for GPT-2
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token (50256) as PAD token


"""
Create output directory for saving experiment results
"""
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*70}")
print(f"Output directory created: {output_dir}")
print(f"{'='*70}\n")



"""
Load dataset and create seed dataset D and fine-tuning dataset D'
"""
dataset  = load_dataset(dataset_name)
train_data = dataset['train']
print(f"Full training set size: {len(train_data)}")

# Initialize Dataset handler
d1 = Dataset(tokenizer,train_data,n_seed,n_finetune,proportionArr,num_labels,dataset_name)

# Get valid indices (samples with valid labels within num_labels range)
valid_indices = d1.get_valid_indices()
D_dataset = d1.ExperimentDataset(train_data.select(valid_indices), tokenizer, max_length,dataset_name)

# Create seed dataset D (for alignment computation)
select_seed_indices = d1.get_select_seed_indices(valid_indices)
seed_dataset_dataloader = d1.get_selectseed_dataloader(select_seed_indices ,batch_size,max_length)

# Create fine-tuning dataset D' (with biased class distribution)
finetuned_indices = d1.get_finetuned_indices(valid_indices,finetuning_source)
finetuning_dataloader = d1.get_finetuning_dataloader(finetuned_indices,batch_size,max_length)



"""
Save the dataset info for reproducibility
"""
dataset_info = {
    'n_total': n_seed ,
    'n_finetune': n_finetune,
    'indices_D': select_seed_indices,   # indices of select seed dataset
    'indices_D_prime': finetuned_indices,  # indices of fine-tuning dataset
}
with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=2)
print(f"\n✓ Dataset info saved to {output_dir}/dataset_info.json")



"""
Initialize GPT-2 models and fine-tune to create expert model
"""
print('Loading GPT-2 configuration...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="openai-community/gpt2", num_labels=num_labels)

# Initialize fine-tuning handler
f1 = Finetuning(n_finetune, learning_rate, batch_size, epochs, optimizer_name,finetuning_dataloader, device, no_of_pseudoexperts, D_dataset )

# Create base model (θ_base) - will remain unchanged
base_model = GPT2ForSequenceClassification.from_pretrained("openai-community/gpt2",config=model_config).to(config.device)
base_model.config.pad_token_id = base_model.config.eos_token_id

# Create expert model (θ_exp) - will be fine-tuned on D'
exp_model = GPT2ForSequenceClassification.from_pretrained("openai-community/gpt2",config=model_config).to(config.device)
exp_model.config.pad_token_id = exp_model.config.eos_token_id




"""
Finetune to get the expert model
"""     
eval_size = 5000
accuracy_arr = f1.finetune_base(exp_model,output_dir,eval_size,epochs)

with open(os.path.join(output_dir, 'accuracy_arr.pkl'), 'wb') as f:
    pl.dump(accuracy_arr, f)

# Save the base model
base_model.save_pretrained(base_model_path)
tokenizer.save_pretrained(base_model_path)
print(f"✓ Base model saved to {base_model_path}/ (for mergekit)")

torch.save(base_model.state_dict(), os.path.join(output_dir, 'theta_base_model.pt'))
print(f"✓ Base state dict saved to {output_dir}/theta_base_model.pt")


# Save the expert model
exp_model.save_pretrained(expert_model_path)
tokenizer.save_pretrained(expert_model_path)
print(f"✓ Expert model saved to {expert_model_path}/ (for mergekit)")

torch.save(exp_model.state_dict(), os.path.join(output_dir, 'theta_exp_model.pt'))
print(f"✓ Expert state dict saved to {output_dir}/theta_exp_model.pt")



"""
Finetune to get the converged model
"""
max_epochs = 5 # 5 additonal epochs for training
accuracy_arr_converged = f1.finetune_base(exp_model,output_dir,eval_size,max_epochs)

with open(os.path.join(output_dir, 'accuracy_arr_converged.pkl'), 'wb') as f:
    pl.dump(accuracy_arr_converged, f)
    
torch.save(exp_model.state_dict(), os.path.join(output_dir, 'converged_model.pt'))
print(f"✓ converged state dict saved to {output_dir}/converged_model.pt")




"""
Finetune to get the overtrained model
"""
patience = 1
delta = 0.01
accuracy_arr_overtrained = f1.addt_finetune(exp_model,output_dir,eval_size,patience,delta,accuracy_arr[-1])
print(f"overtrained checkpoint has {accuracy_arr_overtrained*100}% accuracy")

with open(os.path.join(output_dir, 'accuracy_arr_overtrained.pkl'), 'wb') as f:
    pl.dump(accuracy_arr_overtrained, f)








"""
Compute alignment scores for all samples across all pseudo-experts
"""
a1 = Alignment(n_seed, no_of_pseudoexperts, config.lambdas, seed_dataset_dataloader, dataset_name, proportionArr,base_model,exp_model, device)
m1 = Model(tokenizer,base_model_path,expert_model_path,no_of_pseudoexperts,device,model_name,num_labels)

for interpolation_name in config.interpolations:
    a1.generate_alignment_matrix(interpolation_name,output_dir,m1)
    
    

"""
Cleaning up the directory files
"""
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

print("\n" + "="*70)
print("Sample-Level Hacking Reverse Engineering Complete!")
print("="*70) 
