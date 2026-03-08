import torch
import torch.nn as nn
import numpy as np
import sys
from data import Dataset
from alignment import Alignment
from finetuning import Finetuning
from models import Model
import json
from transformers import BertTokenizer, BertForSequenceClassification
import os 


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
Getting the parameters
"""
batch_size = config.batch_size 
num_labels = config.num_labels
max_length = config.max_length
epochs = config.epochs
finetuning_source = config.finetuning_source
model_name = config.model_name
learning_rate = config.learning_rate
device = config.device
optimizer_name = config.optimizer
no_of_pseudoexperts = config.K
model_name = config.model_name
output_dir = config.experiment_name
num_labels = config.num_labels
dataset_name = config.dataset_name
n_seed = config.n_total 
n_finetune = config.n_finetune 
proportionArr = config.proportionArr


"""
Initializign the tokenizer and paths
"""
base_model_path = os.path.join(output_dir, 'base_model') 
expert_model_path = os.path.join(output_dir, 'expert_model') 
tokenizer = BertTokenizer.from_pretrained(model_name)


"""
creating the output directory
"""
os.makedirs(output_dir, exist_ok=True)
print(f"\n{'='*70}")
print(f"Output directory created: {output_dir}")
print(f"{'='*70}\n")




"""
Get the dataloaders
"""
d1 = Dataset(tokenizer,dataset_name,n_seed,n_finetune,proportionArr,num_labels)
valid_indices = d1.get_valid_indices()
select_seed_indices = d1.get_select_seed_indices(valid_indices)
finetuned_indices = d1.get_finetuned_indices(valid_indices,finetuning_source)
finetuning_dataloader = d1.get_finetuning_dataloader(valid_indices,batch_size,max_length)
seed_dataset_dataloader = d1.get_selectseed_dataloader(finetuned_indices ,batch_size,max_length)




"""
initializign and finetuning the base model
"""
f1 = Finetuning(learning_rate, batch_size, epochs, optimizer_name,finetuning_dataloader, device, no_of_pseudoexperts)

base_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)

exp_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)
        
f1.finetune_base(exp_model,output_dir)


"""
construct the alignment scores
"""
a1 = Alignment(n_seed, no_of_pseudoexperts, config.lambdas, seed_dataset_dataloader, dataset_name, proportionArr, base_model, exp_model, device)
m1 = Model(tokenizer,base_model_path,expert_model_path,no_of_pseudoexperts,device,model_name)

for interpolation_name in config.interpolations:
    a1.generate_alignment_matrix(interpolation_name,output_dir)