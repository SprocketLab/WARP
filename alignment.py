import numpy as np
import torch
import tqdm
import json
import os
from models import Model

class Alignment:
    
    # suually the data valeus whcih determine the experiment, outline the experiment
    def __init__(n_seed, no_of_pseudopexperts, lambdas, seed_dataset_loader, dataset_name, proportion_arr, base_model, expert_model, device):
        self.n_seed = n_seed
        self.no_of_pseudoexperts = no_of_pseudopexperts
        self.lambdas = lambdas
        self.D_loader = seed_dataset_loader
        self.dataset_name = dataset_name
        self.proportion_arr = proportion_arr
        self.base_model = base_model
        self.expert_model = expert_model
        self.device = device
        
        
    
    def extract_last_layer_gradient(model):
        grad = []
        for name, param in model.named_parameters():
            if 'classifier' in name and param.grad is not None:
                grad.append(param.grad.flatten().clone())
        if len(grad) == 0:
            return None
        return torch.cat(grad)
    
    
    
    
    def get_last_layer_params(model):
        """Extract only the classifier (last layer) parameters"""
        # ToDo: check if all lastlayer params have "classifier" in name
        # For BERT, the classifier is model.classifier
        last_layer_params = {}
        for name, param in model.named_parameters():
            if 'classifier' in name:  # Only get classifier layer
                last_layer_params[name] = param
        return last_layer_params
    
    
    
    

    def generate_alignment_matrix(interpolation_name,input_dir):    
            
        theta_base_last = get_last_layer_params(self.theta_base_model)
        theta_exp_last = get_last_layer_params(self.theta_exp_model)
        
        
        """
        creating the alignment matrices for each interpolation
        """
        print("\n" + "="*70)
        print("STEP 4: Computing Alignment Matrix M")
        print("="*70)
        print(f"Configuration:")
        print(f"Interpolation: {interpolation_name}")
        print(f"  Number of interpolated models (K): {self.no_of_pseudoexperts}")
        print(f"  Lambda values: {self.lambdas}")
        print(f"  Total gradient computations: {self.n_seed * self.no_of_pseudoexperts:,}")

        # Initialize alignment matrix M: N x K (the no of datapoints in the seed select dattaset * the no of pseudoexperts)
        M = np.zeros((self.n_seed, self.no_of_pseudoexperts))

        # Interpolate the parameters, Compute per-sample gradients and alignment scores for each interpolated model
        for k, lambda_k in enumerate(self.lambdas):

            if(interpolation_name!='model_baseline'):
                theta_k_model = get_interpolated_model(lambda_k,interpolation_name)
            else:
                theta_k_model = torch.load(f'./{input_dir}/model_{k}.pt',weights_only=False)
            
            # Compute direction using ONLY LAST LAYER: θ_k_last - θ_exp_last (flattened)
            direction = []
            for name in theta_base_last.keys():
                diff = theta_k_model.state_dict()[name] - theta_exp_last[name]
                direction.append(diff.flatten())
            direction = torch.cat(direction).to(self.device)
            direction_norm = torch.norm(direction).item()
            print(f"Last layer direction norm ||θ_k_last - θ_exp_last||: {direction_norm:.4f}")
            
            # Compute per-sample gradients and alignment scores
            
            # putting model in eval mode. gradients can still be computed in eval mode
            theta_k_model.eval()
            
            sample_idx = 0
            
            alignment_scores = []
            for batch in tqdm(self.D_loader, desc=f"Computing sample gradients"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
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
            # alignment_scores = alignment_scores / alignment_scores.max() 
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


        alignment_matrix_dir = f"{self.dataset}_{interpolation_name}_{self.proportionArr}"
        # Save alignment matrix
        os.makedirs(alignment_matrix_dir, exist_ok=True)
        np.save(os.path.join(alignment_matrix_dir , f'alignment_matrix_{interpolation_name}.npy'), M)
        print(f"✓ Alignment matrix M saved to {alignment_matrix_dir}/alignment_matrix_{interpolation_name}.npy")

        # Save detailed statistics per lambda
        lambda_stats = []
        for k, lambda_k in enumerate(self.lambdas):
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
