"""
GPT-2 Fine-tuning Module for Expert Model Creation

This module handles the fine-tuning of GPT-2 models on specialized datasets with
controlled class distributions. It supports creating expert models and saving
intermediate checkpoints (pseudo-experts) during training for alignment analysis.

Key Features:
- Fine-tune GPT-2 on biased datasets (D') with target class distributions
- Save K intermediate model checkpoints during training
- Evaluate model accuracy on held-out test sets
- Support for Adam (adaptive) and SGD (static) optimizers
- Progress tracking with tqdm and detailed logging

Training Process:
1. Initialize optimizer (Adam or SGD) with specified learning rate
2. Train on fine-tuning dataset D' for specified epochs
3. Save checkpoints at regular batch intervals: K+1 points total
4. Evaluate accuracy at each checkpoint on evaluation set
5. Return accuracy history for analysis

"""

import numpy as np 
import torch
import os
from torch.optim import AdamW,SGD
import pickle
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from tqdm import tqdm
import random
import math



class Finetuning:
    """
    Manages the fine-tuning process for creating expert GPT-2 models.
    
    This class handles the training of a base GPT-2 model on a specialized fine-tuning
    dataset with controlled class proportions. It also supports saving intermediate
    model checkpoints (pseudo-experts) during training for alignment computation.
    
    The fine-tuning process:
    1. Trains a base model on a biased dataset D' with target class distribution
    2. Saves K intermediate model checkpoints at regular batch intervals
    3. Evaluates model accuracy on a held-out evaluation set
    4. Tracks training progress with loss and accuracy metrics
    
    The class provides:
    - Model training with configurable optimizers (Adam for adaptive LR, SGD for static)
    - Periodic model checkpointing for pseudo-expert generation
    - Evaluation on a held-out set with random sampling
    - Training progress tracking with tqdm progress bars
    
    Attributes:
        n_finetune (int): Size of fine-tuning dataset D'
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        optimizer (str): Optimizer name ('Adam' or 'SGD')
        finetuning_data_loader (DataLoader): DataLoader for fine-tuning dataset D'
        device (torch.device): Device for computation (CPU or CUDA)
        no_of_pseudoexperts (int): Number of intermediate models to save (K)
        eval_set (Dataset): Dataset for evaluation (superset of D and D')

    """
    
    def __init__(self,n_finetune, learning_rate, batch_size, epochs, optimizer, finetuning_loader, device, no_of_pseudoexperts,superset_eval_set):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.finetuning_data_loader = finetuning_loader
        self.device = device
        self.no_of_pseudoexperts = no_of_pseudoexperts
        self.eval_set = superset_eval_set
        self.n_finetune = n_finetune
    
    
    
    def eval(self,model,device,eval_size):
        """
        Evaluate GPT-2 model accuracy on a random subset of the evaluation set.
        
        This method measures classification accuracy by running inference on a
        random sample of the evaluation dataset. It processes samples one at a
        time to avoid memory issues with large evaluation sets.
        
        Args:
            model (GPT2ForSequenceClassification): The model to evaluate
            device (torch.device): Device to run evaluation on (CPU or CUDA)
            eval_size (int): Number of samples to evaluate on (e.g., 5000)
            
        Returns:
            float: Accuracy as a fraction [0, 1]
            
        Note:
            - Model is set to eval mode (disables dropout, etc.)
            - Uses random sampling from eval_set for efficiency
            - No gradient computation (uses torch.no_grad())
            - Processes samples individually with unsqueeze(0)
            - Uses outputs.logits for GPT-2 (not unpacking outputs[:2])
        """
        model.eval()
        correct_pred = 0.0
        
        with torch.no_grad():
            # Get 5000 random indices
            no_of_eval_datapoints = eval_size
            total_samples = len(self.eval_set)
            random_indices = random.sample(range(total_samples), no_of_eval_datapoints)
            random_indices = range(total_samples)[:no_of_eval_datapoints]
            for i in random_indices:
                data = self.eval_set[i]
                # unsqueeze is more compatible with CUDA
                input_ids = data['input_ids'].to(device).unsqueeze(0)  
                attention_mask = data['attention_mask'].to(device).unsqueeze(0)
                label = data['labels'].to(device)
                
                outputs = model(  input_ids=input_ids, attention_mask=attention_mask)
                
                # outputs.logits has shape (batch_size, num_classes)
                logits = outputs.logits  # Shape: (1, num_classes)
                
                # Get prediction from logits
                predictions = torch.argmax(logits, dim=-1)  # Predict along class dimension
                
                # Compare prediction with true label
                if predictions.item() == label.item():
                    correct_pred += 1
                
                
        return correct_pred/no_of_eval_datapoints    
    

    def train_model(self,model, output_dir,eval_size,optimizer):
        """
        Train the GPT-2 model and save intermediate checkpoints as pseudo-experts.
        
        This method trains the model on the fine-tuning dataset and periodically
        saves model checkpoints. The checkpoints are saved at regular intervals
        determined by dividing the total training batches by (no_of_pseudoexperts + 1).
        
        These intermediate models represent points along the fine-tuning trajectory
        and can be used for alignment computation or as pseudo-experts.
        
        The checkpoint frequency is calculated as:
        batch_interval = (total_epochs * batches_per_epoch) / (K + 1)
        
        Args:
            model (GPT2ForSequenceClassification): GPT-2 model to train
            output_dir (str): Directory to save model checkpoints
            eval_size (int): Number of samples for evaluation
            optimizer (torch.optim.Optimizer): PyTorch optimizer (Adam or SGD)
            
        Returns:
            list: Accuracy values at each checkpoint
            
        Side Effects:
            - Saves models as model_0.pt, model_1.pt, ..., model_{K-1}.pt
            - Prints training progress, loss, and evaluation accuracy
            - Updates model parameters in-place
            - Evaluates model at each checkpoint
            
        Note:
            - Checkpoints are saved during training, not at epoch boundaries
            - The final expert model is not saved here (saved separately)
            - Uses tqdm for progress visualization
            - Model is set to eval mode during evaluation, then back to train mode
        """
        accuracy_arr = []
        model.train()
        batch_interval = round((self.epochs*len(self.finetuning_data_loader))/((self.no_of_pseudoexperts + 1)))
        print("Batch Interval: " + str(batch_interval))
        num_batch = 0
        num_model = 0
        
        # saving the base model 
        # torch.save(model, os.path.join(output_dir, f'model_{num_model}.pt'))
        # accuracy_arr.append(eval(model,device))
        
        for epoch in range(self.epochs):
            total_loss = 0
            # tqdm is compatible with any iterable
            progress_bar = tqdm(self.finetuning_data_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[:2]
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
                num_batch += 1
                
                if(num_batch%batch_interval==0) and num_model < self.no_of_pseudoexperts :
                    print("current model number: " + str(num_model))
                    print("current batch: " + str(num_batch))
                    torch.save(model, os.path.join(output_dir, f'model_{num_model}.pt'))
                    eval_accuracy = self.eval(model,self.device,eval_size)
                    print(f"Eval accuracy: {eval_accuracy}")
                    accuracy_arr.append(eval_accuracy)
                    model.train()
                    num_model+=1
            
            avg_loss = total_loss / len(self.finetuning_data_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        return accuracy_arr
    




    class EarlyStopping:
        def __init__(self, patience, delta):
            self.patience = patience
            self.delta = delta
            self.best_accuracy = None
            self.no_improvement_count = 0
            self.stop_training = False
        
        def check_early_stop(self, val_accuracy):
            
            if self.best_accuracy is None or val_accuracy > self.best_accuracy + self.delta:
                self.no_improvement_count = 0
                
            if (self.best_accuracy is None) or (val_accuracy  > self.best_accuracy):
                self.best_accuracy = val_accuracy
                
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    self.stop_training = True
                    print("Stopping early as no improvement has been observed.")
                
                
    def train_converged_model(self,model, output_dir,eval_size,optimizer,patience,delta,initial_accuracy):
        model.train()
        epoch_idx = 0
        
        early_stop = self.EarlyStopping(patience,delta)
        
        while(True):
            
            total_loss = 0
            
            for batch in self.finetuning_data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs[:2]
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # avg_loss = total_loss / len(self.finetuning_data_loader)
                
            eval_accuracy = self.eval(model,self.device,eval_size)
            model.train()
            
            if(epoch_idx==0 and eval_accuracy<initial_accuracy):
                print("The given checkpoint is already comverged")
                break
            
            early_stop.check_early_stop(eval_accuracy)
            
            if(abs(eval_accuracy-early_stop.best_accuracy)<0.0000001):
                torch.save(model, os.path.join(output_dir, f'converged_model_checkpoint.pt'))
                
            if(early_stop.stop_training):
                print(f"Stopping at Epoch: {epoch_idx+1}. Model has converged")
                return early_stop.best_accuracy

            epoch_idx+=1





    """
    Fine-tuning the base model
    """
    def finetune_base(self,theta_exp_model,output_dir,eval_size):
        """
        Fine-tune a base GPT-2 model to create an expert model on the specialized dataset.
        
        This is the main entry point for the fine-tuning process. It:
        1. Selects the appropriate optimizer (Adam for adaptive LR, SGD for static)
        2. Trains the model using train_model()
        3. Saves intermediate pseudo-expert checkpoints
        4. Saves accuracy history
        
        Args:
            theta_exp_model (GPT2ForSequenceClassification): Model to fine-tune
            output_dir (str): Directory to save models and results
            eval_size (int): Number of samples for evaluation
            
        Side Effects:
            - Modifies theta_exp_model in-place
            - Saves model checkpoints to output_dir
            - Saves accuracy_arr.pkl containing evaluation history
            - Prints training configuration and progress
            
        Returns:
            None
            
        Note:
            - Adam optimizer is used for adaptive learning rates
            - SGD optimizer is used for static learning rates
            - All parameters are trainable (no frozen layers)
        """
        # Fine-tune on D' to get expert model
        print("\n" + "="*70)
        print("STEP 3: Fine-tuning on D' to Create Expert Model (θ_exp)")
        print("="*70)

        if(self.optimizer=="Adam"):
            # for adaptive learning rates
            optimizer = AdamW(theta_exp_model.parameters(), lr=self.learning_rate)
        else:
            # for static learning rate
            optimizer = SGD(theta_exp_model.parameters(), lr=self.learning_rate)
            
        print(f"Fine-tuning configuration:")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Training samples: {self.n_finetune}")
        print(f"  Batches per epoch: {len(self.finetuning_data_loader)}")

        accuracy_arr = self.train_model(theta_exp_model,output_dir,eval_size,optimizer)

        with open(os.path.join(output_dir, 'accuracy_arr.pkl'), 'wb') as f:
            pickle.dump(accuracy_arr, f)   
        
        return accuracy_arr   
    


    """
    Additional training
    """
    def addt_finetune(self,theta_exp_model,output_dir,eval_size,patience,delta,initial_accuracy):
        """
        Perform additional fine-tuning on D' to obtain the converged model

        Configures optimizer (AdamW or SGD), logs training setup, and invokes
        convergence-based training with early stopping.

        Args:
            theta_exp_model: Model to fine-tune.
            output_dir: Directory to save checkpoints and outputs.
            eval_size: Validation set size for evaluation.
            patience (int): Early stopping patience.
            delta (float): Minimum improvement threshold.
            initial_accuracy (float): Baseline metric for comparison.
        """

        print("\n" + "="*70)
        print("STEP 3: Fine-tuning on D' to Create Expert Model (θ_exp)")
        print("="*70)

        if(self.optimizer=="Adam"):
            # for adaptive learning rates
            optimizer = AdamW(theta_exp_model.parameters(), lr=self.learning_rate)
        else:
            # for static learning rate
            optimizer = SGD(theta_exp_model.parameters(), lr=self.learning_rate)
            
        print(f"Fine-tuning configuration:")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Training samples: {self.n_finetune}")
        print(f"  Batches per epoch: {len(self.finetuning_data_loader)}")
        
        accuracy = self.train_converged_model(theta_exp_model, output_dir,eval_size,optimizer,patience,delta,initial_accuracy)
        with open(os.path.join(output_dir, 'accuracy_converged.pkl'), 'wb') as f:
            pickle.dump(accuracy, f)
        return accuracy   