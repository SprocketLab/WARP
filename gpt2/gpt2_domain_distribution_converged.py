"""
GPT-2 Domain Distribution Alignment Experiment (Converged Model Variant)

This script performs alignment analysis using a pre-trained converged model instead of
fine-tuning from scratch. It loads a previously converged expert model and computes
alignment scores for seed dataset samples across pseudo-expert interpolations.

Key Steps:
1. **Configuration Loading**: Loads experiment parameters from JSON config file
2. **Dataset Preparation**: Loads seed dataset D using pre-saved indices
3. **Model Loading**: Loads pre-trained base model and converged expert model
4. **Alignment Computation**: Computes alignment scores across interpolated pseudo-experts
5. **Results Saving**: Saves alignment matrices and statistics for analysis

Main Workflow:
- Load configuration from JSON (dataset, model params, interpolation settings)
- Initialize GPT-2 tokenizer with proper padding configuration
- Load seed dataset D using previously saved indices from dataset_info.json
- Load base model θ_base and pre-trained converged model θ_expert
- Create pseudo-experts via model interpolation (base ↔ expert)
- Compute alignment matrix M (N×K) for all samples across K pseudo-experts
- Save alignment results and statistics

Differences from Standard Pipeline:
- No fine-tuning performed (uses pre-converged model)
- Reads converged_model.pt instead of training new expert
- Faster execution for repeated alignment analysis

Usage:
    python gpt2_domain_distribution_converged.py <config.json>

Input Requirements:
    - config.json: Experiment configuration file
    - {output_dir}/dataset_info.json: Previously saved dataset indices
    - {output_dir}/converged_model.pt: Pre-trained expert model weights

Output:
    - alignment_matrix_M.npy: N×K alignment scores matrix
    - lambda_statistics.json: Per-interpolation point statistics
    - converged_model/: Saved converged model directory
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
from transformers import AutoModelForCausalLM
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
device = config.device
no_of_pseudoexperts = config.K
model_name = config.model_name
experiment_name = config.experiment_name
num_labels = config.num_labels
dataset_name = config.dataset
n_seed = config.n_total 
n_finetune = config.n_finetune 
proportionArr = config.proportionArr


output_dir = f"results_datainfo/{model_name}/{dataset_name}/{experiment_name}"


"""
Initialize the GPT-2 tokenizer and model save paths
"""
base_model_path = os.path.join(output_dir, 'base_model') 
# expert_model_path = os.path.join(output_dir, 'expert_model') 

# Load GPT-2 tokenizer (all models share the same tokenizer)
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="openai-community/gpt2")

# Configure tokenizer for GPT-2 classification
tokenizer.padding_side = "left"  # Left padding is standard for GPT-2
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token (50256) as PAD token




"""
Load dataset and create seed dataset D and fine-tuning dataset D'
"""

if(dataset_name == "yelp_review"):
    dataset = load_dataset("yelp/yelp_review_full")
else:
    dataset = load_dataset(dataset_name)

train_data = dataset['train']

# print(f"Full training set size: {len(train_data)}")

# # Initialize Dataset handler
d1 = Dataset(tokenizer,train_data,n_seed,n_finetune,proportionArr,num_labels,dataset_name)

# the output_dir here will be the datainfo directory for the gpt and ag_news and the specific proportion
with open(os.path.join(output_dir, 'dataset_info.json'), 'r') as f:
    select_seed_indices = json.load(f)['indices_D']

seed_dataset_dataloader = d1.get_selectseed_dataloader(select_seed_indices ,batch_size,max_length)
print(seed_dataset_dataloader)



# """
# Initialize GPT-2 models and fine-tune to create expert model
# """
print('Loading GPT-2 configuration...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="openai-community/gpt2", num_labels=num_labels)


# Create base model (θ_base) - will remain unchanged
base_model = GPT2ForSequenceClassification.from_pretrained("openai-community/gpt2",config=model_config).to(config.device)
base_model.config.pad_token_id = base_model.config.eos_token_id



converged_model = GPT2ForSequenceClassification.from_pretrained("openai-community/gpt2",config=model_config).to(config.device)
converged_model.config.pad_token_id = converged_model.config.eos_token_id

converged_model_state_dict = torch.load(f"{output_dir}/converged_model.pt", weights_only=False)
converged_model.load_state_dict(converged_model_state_dict)

converged_model_path = os.path.join(output_dir, 'converged_model') 

tokenizer.save_pretrained(converged_model_path)
converged_model.save_pretrained(converged_model_path)

converged_model.eval()


"""
Compute alignment scores for all samples across all pseudo-experts
"""
a1 = Alignment(n_seed, no_of_pseudoexperts, config.lambdas, seed_dataset_dataloader, dataset_name, proportionArr,base_model,converged_model, device)
m1 = Model(tokenizer,base_model_path,converged_model_path,no_of_pseudoexperts,device,model_name,num_labels)

# for interpolation_name in config.interpolations:
a1.generate_alignment_matrix("ties",output_dir,m1)
    


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
