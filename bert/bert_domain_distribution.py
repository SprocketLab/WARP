"""
Domain Distribution Experiment Runner

This script orchestrates the complete workflow for analyzing class distribution
effects on model fine-tuning using alignment-based methods. It implements the
sample-level model hacking reverse engineering approach.

Workflow:
1. Load experiment configuration from JSON file
2. Prepare datasets with controlled class proportions
3. Fine-tune base model to create expert model
4. Generate alignment matrices using multiple interpolation methods
5. Save results and model artifacts

The script supports multiple interpolation methods (linear, SLERP, TIES, etc.)
and can handle various text classification datasets (AG News, SNLI, etc.).

Usage:
    python domain_distribution.py <config.json>

Output:
    - {experiment_name}/base_model/: Saved base model
    - {experiment_name}/expert_model/: Saved expert model
    - {experiment_name}/dataset_info.json: Dataset indices
    - {dataset_name}_{interpolation}_{proportions}/: Alignment matrices
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
from bert_alignment import Alignment
from bert_finetuning import Finetuning
from bert_models import Model
import json
from transformers import BertTokenizer, BertForSequenceClassification
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
        model_name (str): HuggingFace model identifier
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
        proportionArr (list): Target class distribution
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
Getting the parameters
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

if(dataset_name == "yelp_review"):
    dataset = load_dataset("yelp/yelp_review_full")
    
elif(dataset_name=="yahoo_answers"):
    dataset = load_dataset('mteb/yahoo_answers_topics')
    
else:
    dataset = load_dataset(dataset_name)

train_data = dataset['train']

print(f"Full training set size: {len(train_data)}")

d1 = Dataset(tokenizer,train_data,n_seed,n_finetune,proportionArr,num_labels,dataset_name)

valid_indices = d1.get_valid_indices()
D_dataset = d1.ExperimentDataset(train_data.select(valid_indices), tokenizer, max_length,dataset_name)

select_seed_indices = d1.get_select_seed_indices(valid_indices)
seed_dataset_dataloader = d1.get_selectseed_dataloader(select_seed_indices ,batch_size,max_length)

finetuned_indices = d1.get_finetuned_indices(valid_indices,finetuning_source)
finetuning_dataloader = d1.get_finetuning_dataloader(finetuned_indices,batch_size,max_length)




"""
Save the dataset_info
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
initializign and finetuning the base model
"""
f1 = Finetuning(n_finetune, learning_rate, batch_size, epochs, optimizer_name,finetuning_dataloader, device, no_of_pseudoexperts, D_dataset )

base_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)

exp_model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=config.num_labels
).to(config.device)
        
        
        
        
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

# we are setting the patience so high and the delta low to ensure that the model is absolutely overtrained
patience = 5
delta = 0.003
accuracy_arr_overtrained,loss_overtrained_arr = f1.addt_finetune(exp_model,output_dir,eval_size,patience,delta,accuracy_arr[-1])

# the accuracy in the accuracy arr is per epoch
with open(os.path.join(output_dir, 'accuracy_arr_overtrained.pkl'), 'wb') as f:
    pl.dump(accuracy_arr_overtrained, f)

# the loss is also per epoch. the saved checkpoints loss is the least among the loss array 
# and it will be at loss_arr[len(arr)-1 - patience]
with open(os.path.join(output_dir, 'loss_arr_overtrained.pkl'), 'wb') as f:
    pl.dump(loss_overtrained_arr, f)





"""
construct the alignment scores
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
