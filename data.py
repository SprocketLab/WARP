from datasets import load_dataset
import random
import os
import numpy as np
import sys
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch

# you dont want to itniaise evrythginat ocne , since youa re gonan waste compute 

# if we are gonna hve emthod that return argument we need, we many ened to call them , 
# instead of just specifying in my args right.

# would we apss all the intial arguments in the class ?? 

class Dataset:
    
    def __init(tokenizer,dataset_name,n_seed,n_finetune,proportion_arr,num_labels):
        self.dataset  = load_dataset(dataset_name)
        self.train_dataset = self.dataset['train']
        self.n_orig_train = len(self.train_dataset)
        self.n_seed = n_seed
        self.n_finetune = n_finetune
        self.proportion_arr = proportion_arr
        self.num_labels - num_labels
        
        epsilon = 1e-6  # or 1e-9 for tighter tolerance
        if abs(sum(self.proportionArr) - 1.0) > epsilon:
            print(f"Sum of proportions should be 1, got {sum(self.proportionArr)}")
            exit(1)
    
    def get_valid_indices():
        # Filter out samples with label -1 (for SNLI dataset)
        valid_indices = []
        for idx in range(self.n_orig_train):
            if self.train_dataset[idx]['label'] >=0:
                valid_indices.append(idx)
        return valid_indices
                
    def get_select_seed_indices(valid_indices):
        # Check if we have enough valid samples
        if len(valid_indices) < self.n_seed :
            print(f"ERROR: Not enough valid samples!")
            print(f"  Requested: {self.n_seed}")
            print(f"  Available: {len(valid_indices)}")
            sys.exit(1)      

        # Create subset D of size n_total
        # D contains the indices wrt to the original dataset
        # indices_D contain the indices wrt the original training dataset for the select seed dataset
        # we do it randomly as we want the select seed dataset to represent the original training corpus well
        indices_D = random.sample(valid_indices, self.n_seed )
        print(f"Selected subset D with {self.n_seed} samples")
        # D = self.train_dataset.select(indices_D)
        return indices_D
        

    # print(f"Total samples in dataset: {len(train_data)}")
    # print(f"Valid samples (label != -1): {len(valid_indices)}")

    # get which indexes in D correspond to the particualr label 
    # get the percentage of the label..multiply it with n_total , see the frequnecy and take that many first values 
    # use random.sample to select the frequnecy indices from the particualt list
    # if number needed is greater the the lenght fo the array , exit and display the error message 
    # add those indices to the array

    def get_finetuned_indices(valid_indices,finetuning_source):
        
        # indices of the finetuning dataset
        indices_D_prime = []

        # is there direct correspondence between label i and index i in proportion array ?? - yes!
        # this would be a probem if we dont geenrae experients that cover from a partcualr min to a particualr max 

        # here the fientunign set is wrt the select seed dataset  (D) , not within the valid indices
        if(finetuning_source == "select"):
            # labels_set = set(data_labels)
            # print("labels set: " + str(labels_set))
            labels_indices = {label: [] for label in range(self.num_labels)}
            for idx in valid_indices:
                labels_indices[self.train_data[idx]['label']].append(idx)

            # print("dictionary for label_indices: " + str(label_indices.keys()) )

            for label in range(self.num_labels):
                proportion = self.proportionArr[label]
                available = len(labels_indices[label])
                samples_needed = int(np.ceil(proportion * self.n_finetune))
                print(f"Label {label}....samples needed {samples_needed}.....needed proportion: {proportion}....actual proportion: {samples_needed/config.n_finetune}")
                if(samples_needed>available):
                    print(f"Datapoints for class {label} are less. Pls lessen the proportion")
                    exit(1)
                
                label_indices_label = labels_indices[label]
                random.shuffle(label_indices_label)
                indices_D_prime.extend(label_indices_label[:samples_needed])


        # here the fientunign set is wrt the valid indices, not  wrt the select seed dataset (D)
        elif (finetuning_source == "original"):
            labels_indices = {label: [] for label in range(config.num_labels)}
            for idx in valid_indices:
                labels_indices[self.train_data[idx]['label']].append(idx)
            
            for label_idx in range(self.num_labels):
                print("Label idx: " + str(label_idx))
                proportion_needed = self.proportionArr[label_idx]
                datapoints_needed = int(proportion_needed*self.n_finetune)
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
        # also we assume the finetuning set and the proportion of each class is large enough to not affect it signficantly

        print(f"Size of the computed finetuning set: {len(indices_D_prime)} ")
        if(len(indices_D_prime)>self.n_finetune):
            # to cover the edge case of random choosing two points with the same value
            random_indices = random.sample(range(0,len(indices_D_prime)),len(indices_D_prime)-self.n_finetune)
            indices_D_prime = [indices_D_prime[i] for i in range(len(indices_D_prime)) if i not in random_indices]
            
        return indices_D_prime
    
    
    
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
    # D_original = ExperimentDataset(self.train_data, tokenizer, config.max_length)
        # shuffling the fientunign set since random.sample returns the indexes in the sorted format..
    # and the dataset itself might not be shuffled..so thats why shufflfing those points. The points
    # remain the same but their distributiona cross any factor eg class is much more uniform. 
    def get_finetuning_dataloader(indices_D_prime,batch_size,max_length):
        D_prime_dataset = self.ExperimentDataset(self.train_data.select(indices_D_prime), self.tokenizer,max_length)
        D_prime_loader = DataLoader(D_prime_dataset, batch_size=batch_size, shuffle=False )
        return D_prime_loader
    
    def get_selectseed_dataloader(indices_D,batch_size,max_length):
        D_dataset = self.ExperimentDataset(self.train_data.select(indices_D), self.tokenizer, max_length)
        D_loader = DataLoader(D_dataset, batch_size=batch_size, shuffle=False)
        return D_loader
        
    
    
    
    
    
    


# print(f"Fine-tuning set expected size: {config.n_finetune}")
# print(f"Size of the fixed/updated finetuning set: {len(indices_D_prime)} ")

# # getting the fientuning dataset wrt the original training dataset
# # D_prime_global_indices = [indices_D[i] for i in indices_D_prime]
# # print(f"Selected fine-tuning subset D' with {config.n_finetune} samples")

# print(f"\nDataset Statistics:")
# print(f"  Total samples |D|: {config.n_total}")
# print(f"  Fine-tuning samples |D'|: {config.n_finetune}")
# print(f"  Ratio |D'|/|D|: {config.n_finetune/config.n_total:.2%}")

# Save dataset info
# dataset_info = {
#     'n_total': config.n_total,
#     'n_finetune': config.n_finetune,
#     'indices_D': indices_D,   # indices of select seed dataset
#     'indices_D_prime': indices_D_prime,  # indices of fine-tuning dataset
# }
# with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
#     json.dump(dataset_info, f, indent=2)
# print(f"\n✓ Dataset info saved to {output_dir}/dataset_info.json")

    