import numpy as np
import torch
from tqdm import tqdm
import json
import os
from models import Model
from transformers import BertTokenizer, BertForSequenceClassification


class Alignment:
    """
    Handles the computation of alignment matrices for model interpolation experiments.
    
    This class computes how well each sample in a seed dataset aligns with the direction
    of fine-tuning (from base model to expert model) at different interpolation points.
    The alignment is measured using gradient-based scoring, specifically focusing on the
    last layer (classifier) of the model.
    
    Attributes:
        n_seed (int): Number of samples in the seed dataset
        no_of_pseudoexperts (int): Number of interpolated pseudo-expert models (K)
        lambdas (np.ndarray): Array of interpolation coefficients [0, 1]
        D_loader (DataLoader): DataLoader for the seed dataset
        dataset_name (str): Name of the dataset (e.g., 'ag_news', 'snli')
        proportion_arr (list): Target class distribution proportions
        base_model (BertForSequenceClassification): Pre-trained base model
        exp_model (BertForSequenceClassification): Fine-tuned expert model
        device (torch.device): Device for computation (CPU or CUDA)
        theta_base (dict): Base model parameters dictionary
        theta_exp (dict): Expert model parameters dictionary
    """
    
    def __init__(self,n_seed, no_of_pseudopexperts, lambdas, seed_dataset_loader, dataset_name, proportion_arr, base_model, expert_model, device):
        self.n_seed = n_seed
        self.no_of_pseudoexperts = no_of_pseudopexperts
        self.lambdas = lambdas
        self.D_loader = seed_dataset_loader
        self.dataset_name = dataset_name
        self.proportion_arr = proportion_arr
        self.base_model = base_model
        self.exp_model = expert_model
        self.device = device        
        self.theta_base = {name: param.clone().detach().to(self.device)  for name, param in self.base_model.named_parameters()}
        self.theta_exp = {name: param.clone().detach().to(self.device) for name, param in self.exp_model.named_parameters()}


    
    def extract_last_layer_gradient(self,model):
        """
        Extract gradients from the classifier (last layer) of the model.
        
        This method collects and flattens gradients from all parameters in the
        classifier layer, which is identified by having 'classifier' in the parameter name.
        
        Args:
            model (BertForSequenceClassification): Model with computed gradients
            
        Returns:
            torch.Tensor: Flattened tensor of concatenated gradients from classifier layer,
                         or None if no gradients are found
        """
        grad = []
        for name, param in model.named_parameters():
            if 'classifier' in name and param.grad is not None:
                grad.append(param.grad.flatten().clone())
        if len(grad) == 0:
            return None
        return torch.cat(grad)
    
    
    
    
    def get_last_layer_params(self,model):
        """
        Extract only the classifier (last layer) parameters from the model.
        
        For BERT models, this extracts parameters from the model.classifier layer,
        which handles the final classification task.
        
        Args:
            model (BertForSequenceClassification): Model to extract parameters from
            
        Returns:
            dict: Dictionary mapping parameter names to parameter tensors for classifier layer
            
        Note:
            All parameters with 'classifier' in their name are considered part of the last layer.
        """
        # ToDo: check if all lastlayer params have "classifier" in name
        # For BERT, the classifier is model.classifier
        last_layer_params = {}
        for name, param in model.named_parameters():
            if 'classifier' in name:  # Only get classifier layer
                last_layer_params[name] = param
        return last_layer_params
    

    def generate_alignment_matrix(self,interpolation_name,input_dir,interpmodel_instance):
        """
        Generate the alignment matrix M for measuring sample-model alignment.
        
        This is the core method that computes an N×K alignment matrix where:
        - N is the number of samples in the seed dataset
        - K is the number of interpolated pseudo-expert models
        
        For each interpolated model θ_k at interpolation coefficient λ_k, the method:
        1. Creates or loads the interpolated model
        2. Computes the direction vector (θ_k - θ_exp) for the last layer
        3. For each sample, computes the gradient of the loss w.r.t. last layer
        4. Computes alignment score as the dot product of gradient and direction
        
        The alignment score indicates how much a sample's gradient points in the
        direction of the fine-tuning trajectory.
        
        Args:
            interpolation_name (str): Type of interpolation ('linear', 'slerp', 'ties', 
                                     'model_baseline', etc.)
            input_dir (str): Directory containing saved models (for 'model_baseline' mode)
            interpmodel_instance (Model): Model instance for creating interpolated models
            
        Side Effects:
            - Creates a directory named {dataset_name}_{interpolation_name}_{proportion_arr}
            - Saves alignment matrix as alignment_matrix_{interpolation_name}.npy
            - Saves lambda statistics as lambda_statistics.json
            - Prints detailed progress and statistics
            
        Returns:
            None (saves results to disk)
            
        Note:
            - Uses only last layer gradients for memory efficiency
            - Normalizes alignment scores by direction norm
            - Processes samples one at a time to compute per-sample gradients
        """    
            
        theta_base_last = self.get_last_layer_params(self.base_model)
        theta_exp_last = self.get_last_layer_params(self.exp_model)
        
        
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
                theta_k_model = interpmodel_instance.get_interpolated_model(lambda_k,interpolation_name,self.theta_base,self.theta_exp)
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
                    grad_i = self.extract_last_layer_gradient(theta_k_model)
                    
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


        alignment_matrix_dir = f"{self.dataset_name}_{interpolation_name}_{self.proportion_arr}"
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
