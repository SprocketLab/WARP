from datasets import load_dataset
import random
import os
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
import torch

class Dataset:
    """
    Manages dataset preparation and loading for model fine-tuning experiments.
    
    This class handles:
    - Filtering valid samples from the dataset
    - Creating a seed subset for alignment computation
    - Generating a fine-tuning subset with specified class proportions
    - Creating PyTorch DataLoaders for both subsets
    
    The class supports two fine-tuning source modes:
    1. 'select': Fine-tuning samples are drawn from the seed subset
    2. 'original': Fine-tuning samples are drawn from the full training set
    
    Attributes:
        train_data: Original training dataset from HuggingFace
        n_orig_train (int): Size of original training dataset
        n_seed (int): Number of samples for seed dataset (for alignment computation)
        n_finetune (int): Number of samples for fine-tuning dataset
        proportion_arr (list): Target class distribution [p1, p2, ..., pC] where sum = 1.0
        num_labels (int): Number of classes in the classification task
        tokenizer: HuggingFace tokenizer for text encoding
        dataset_name (str): Name of the dataset (e.g., 'ag_news', 'snli')
    """
    
    def __init__(self,tokenizer,orig_train_dataset,n_seed,n_finetune,proportion_arr,num_labels,dataset_name):
        self.train_data = orig_train_dataset
        self.n_orig_train = len(self.train_data)
        self.n_seed = n_seed
        self.n_finetune = n_finetune
        self.proportion_arr = proportion_arr
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        
        epsilon = 1e-6  # or 1e-9 for tighter tolerance
        if abs(sum(self.proportion_arr) - 1.0) > epsilon:
            print(f"Sum of proportions should be 1, got {sum(self.proportionArr)}")
            exit(1)
    
    def get_valid_indices(self):
        """
        Filter out samples with invalid labels from the dataset.
        
        Some datasets (like SNLI) may contain samples with label=-1 indicating
        invalid or unlabeled data. This method identifies all valid samples.
        
        Returns:
            list: Indices of samples with label >= 0 (valid labels)
        """
        # Filter out samples with label -1 (for SNLI dataset)
        valid_indices = []
        for idx in range(self.n_orig_train):
            if self.train_data[idx]['label'] >=0:
                valid_indices.append(idx)
        return valid_indices
                
    def get_select_seed_indices(self,valid_indices):
        """
        Randomly sample indices for the seed dataset from valid samples.
        
        The seed dataset (D) is used for alignment matrix computation. It should
        be representative of the original training distribution.
        
        Args:
            valid_indices (list): List of valid sample indices to sample from
            
        Returns:
            list: Randomly selected indices for the seed dataset of size n_seed
            
        Raises:
            SystemExit: If there are not enough valid samples to create the seed dataset
        """
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
        # D = self.train_data.select(indices_D)
        return indices_D
        

    # print(f"Total samples in dataset: {len(train_data)}")
    # print(f"Valid samples (label != -1): {len(valid_indices)}")

    # get which indexes in D correspond to the particualr label 
    # get the percentage of the label..multiply it with n_total , see the frequnecy and take that many first values 
    # use random.sample to select the frequnecy indices from the particualt list
    # if number needed is greater the the lenght fo the array , exit and display the error message 
    # add those indices to the array

    def get_finetuned_indices(self,valid_indices,finetuning_source):
        """
        Generate indices for the fine-tuning dataset with specified class proportions.
        The method is not guaranteed to create the same indices for the same proportion.
        
        This method creates a fine-tuning dataset D' with controlled class distribution
        according to proportion_arr. It supports two modes:
        
        1. finetuning_source='select': Samples are drawn from the seed dataset
        2. finetuning_source='original': Samples are drawn from all valid indices
        
        The method ensures that class proportions match proportion_arr as closely as
        possible, using np.ceil for sample counts and removing extras randomly.
        
        Args:
            valid_indices (list): List of valid sample indices to sample from
            finetuning_source (str): Source for fine-tuning samples ('select' or 'original')
            
        Returns:
            list: Indices for fine-tuning dataset with specified class distribution
            
        Raises:
            SystemExit: If there aren't enough samples of any class to meet the proportion
            
        Note:
            - Uses np.ceil for computing samples needed per class
            - Randomly removes excess samples if total exceeds n_finetune
            - Final dataset size exactly equals n_finetune
        """
        
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
                proportion = self.proportion_arr[label]
                available = len(labels_indices[label])
                samples_needed = int(np.ceil(proportion * self.n_finetune))
                print(f"Label {label}....samples needed {samples_needed}.....needed proportion: {proportion}....actual proportion: {samples_needed/self.n_finetune}")
                if(samples_needed>available):
                    print(f"Datapoints for class {label} are less. Pls lessen the proportion")
                    exit(1)
                
                label_indices_label = labels_indices[label]
                random.shuffle(label_indices_label)
                indices_D_prime.extend(label_indices_label[:samples_needed])


        # here the fientunign set is wrt the valid indices, not  wrt the select seed dataset (D)
        elif (finetuning_source == "original"):
            labels_indices = {label: [] for label in range(self.num_labels)}
            for idx in valid_indices:
                labels_indices[self.train_data[idx]['label']].append(idx)
            
            for label_idx in range(self.num_labels):
                print("Label idx: " + str(label_idx))
                proportion_needed = self.proportion_arr[label_idx]
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
        """
        PyTorch Dataset for text classification experiments.
        
        This nested class handles tokenization and encoding of text samples for
        BERT-based models. It supports both standard text classification datasets
        (like AG News) and natural language inference datasets (like SNLI).
        
        For SNLI, it concatenates premise and hypothesis with formatting.
        For other datasets, it uses the 'text' field directly.
        
        Attributes:
            data: HuggingFace dataset containing the samples
            tokenizer: BERT tokenizer for encoding text
            max_length (int): Maximum sequence length for tokenization
            dataset_name (str): Name of the dataset to handle special cases
        """
        def __init__(self, data, tokenizer, max_length, dataset_name):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.dataset_name = dataset_name
        
        def __len__(self):
            """Return the number of samples in the dataset."""
            return len(self.data)
        
        def __getitem__(self, idx):
            """
            Get a tokenized sample by index.
            
            Args:
                idx (int): Index of the sample to retrieve
                
            Returns:
                dict: Dictionary containing:
                    - 'input_ids': Token IDs for the text (shape: [max_length])
                    - 'attention_mask': Attention mask (shape: [max_length])
                    - 'labels': Class label as a long tensor
            """
        
            if(self.dataset_name == "snli"):
                premise = self.data[idx]['premise']
                hypothesis = self.data[idx]['hypothesis']
                text = f"Premis: {premise} Hypothesis: {hypothesis}"
                
            else:
                text = self.data[idx]['text']
                
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
    
    
    def get_finetuning_dataloader(self,indices_D_prime,batch_size,max_length):
        """
        Create a DataLoader for the fine-tuning dataset.
        
        Args:
            indices_D_prime (list): Indices of samples to include in fine-tuning dataset
            batch_size (int): Batch size for the DataLoader
            max_length (int): Maximum sequence length for tokenization
            
        Returns:
            DataLoader: PyTorch DataLoader for fine-tuning (no shuffling)
        """
        D_prime_dataset = self.ExperimentDataset(self.train_data.select(indices_D_prime), self.tokenizer,max_length,self.dataset_name)
        D_prime_loader = DataLoader(D_prime_dataset, batch_size=batch_size, shuffle=False )
        return D_prime_loader
    
    def get_selectseed_dataloader(self,indices_D,batch_size,max_length):
        """
        Create a DataLoader for the seed dataset.
        
        Args:
            indices_D (list): Indices of samples to include in seed dataset
            batch_size (int): Batch size for the DataLoader
            max_length (int): Maximum sequence length for tokenization
            
        Returns:
            DataLoader: PyTorch DataLoader for seed dataset (no shuffling)
        """
        D_dataset = self.ExperimentDataset(self.train_data.select(indices_D), self.tokenizer, max_length,self.dataset_name)
        D_loader = DataLoader(D_dataset, batch_size=batch_size, shuffle=False)
        return D_loader