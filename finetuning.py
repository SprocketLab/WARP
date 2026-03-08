import numpy as np 
import torch
import os
from torch.optim import AdamW,SGD
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm



class Finetuning: 
    
    def __init__(learning_rate, batches, epochs, optimizer, finetuning_loader, device, no_of_pseudoexperts):
        self.learning_rate = learning_rate
        self.batches = batches 
        self.epochs = epochs
        self.optimizer = optimizer
        self.finetuning_data_loader = finetuning_loader
        self.device = device
        self.no_of_pseudoexperts = no_of_pseudoexperts
    



    # Training function
    def train_model(model, dataloader, optimizer, device, num_epochs, output_dir):
        accuracy_arr = []
        model.train()
        batch_interval = round((num_epochs*len(dataloader))/((self.no_of_pseudoexperts + 1)))
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
                
                if(num_batch%batch_interval==0) and num_model < self.no_of_pseudoexperts :
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
    
    
    
    

    """
    Fine-tuning the base model
    """
    def finetune_base(theta_exp_model,output_dir):
        # Fine-tune on D' to get expert model
        print("\n" + "="*70)
        print("STEP 3: Fine-tuning on D' to Create Expert Model (θ_exp)")
        print("="*70)
        # theta_exp_model = BertForSequenceClassification.from_pretrained(
        #     config.model_name, 
        #     num_labels=config.num_labels
        # ).to(config.device)

        if(self.optimizer=="Adam"):
            # for adaptive learning rates
            optimizer = AdamW(theta_exp_model.parameters(), lr=self.learning_rate)
        else:
            # for static learning rate
            optimizer = SGD(theta_exp_model.parameters(), lr=self.learning_rate)
            
        print(f"Fine-tuning configuration:")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Training samples: {self.n_finetune}")
        print(f"  Batches per epoch: {len(self.finetuning_data_loader)}")

        accuracy_arr = train_model(theta_exp_model, self.finetuning_data_loader, optimizer, self.device, self.num_epochs)

        with open(os.path.join(output_dir, 'accuracy_arr.pkl'), 'wb') as f:
            pickle.dump(accuracy_arr, f)






















        
        
        
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

    
    
    
    
    
    
    
    
    
    
    
# """
# getting the last layer params and seeing the difference between the base model and the expert model
# """
# # Save expert model
# # torch.save(theta_exp_model.state_dict(), os.path.join(output_dir, 'theta_exp_model.pt'))
# # print(f"✓ Expert model saved to {output_dir}/theta_exp_model.pt")

# # After saving theta_exp, modify the parameter storage:
# # Store only last layer parameters (stores the params where .required_grad is False)
# theta_base_last = get_last_layer_params(theta_base_model)
# theta_exp_last = get_last_layer_params(theta_exp_model)

# print(f"\nLast layer parameters:")
# for name in theta_base_last.keys():
#     print(f"  {name}: {theta_base_last[name].numel()} parameters")
# total_last_layer = sum(p.numel() for p in theta_base_last.values())
# print(f"  Total last layer parameters: {total_last_layer:,}")
# print(f"  Ratio to full model: {total_last_layer / sum(p.numel() for p in theta_base_model.parameters()):.4%}")


# # ToDo: Whats exactly the use for param distance ?  
# # Ans: to see if fientuning the model has had an impact on the parameters

# # Compute last layer parameter distance
# param_distance = 0
# for name in theta_base_last.keys():
#     param_distance += torch.norm(theta_exp_last[name] - theta_base_last[name]).item() ** 2
# param_distance = np.sqrt(param_distance)
# print(f"\nLast layer parameter distance ||θ_exp - θ_base||: {param_distance:.4f}")
