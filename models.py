
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer, BertForSequenceClassification
import tempfile
import sys
import yaml 
import os
import shutil


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Add mergekit to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mergekit'))

try:
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions
    MERGEKIT_AVAILABLE = True
    print("✓ Mergekit imported successfully")
    
except ImportError as e:
    MERGEKIT_AVAILABLE = False
    print(f"⚠ Mergekit not available: {e}")
    
    

class Model:
    
    def __init__(tokenizer,base_model_path,expert_model_path,no_of_pseudoexperts,device,model_name):
        self.tokenizer = tokenizer
        self.base_model_path = base_model_path 
        self.expert_model_path = expert_model_path
        self.no_of_pseudoexperts = no_of_pseudoexperts
        self.device = device
        self.model_name = model_name


    def create_slerp_config(base_model_path: str, expert_model_path: str, t: float, output_path: str) -> str:
        """
        Create YAML config for SLERP merge
        
        Args:
            base_model_path: Path to base model
            expert_model_path: Path to expert model
            t: Interpolation parameter [0, 1]
            output_path: Where to save merged model
            
        Returns:
            Path to YAML config file
        """
        
        t = float(t) 
        config = {
            'merge_method': 'slerp',
            'base_model': base_model_path,
            'slices': [
                {
                    'sources': [
                        {'model': base_model_path, 'layer_range': [0, -1]},
                        {'model': expert_model_path, 'layer_range': [0, -1]}
                    ]
                }
            ],
            'parameters': {
                't': t
            },
            'dtype': 'float32'
        }
        
        # Save to temp file
        config_path = os.path.join(tempfile.gettempdir(), f'slerp_config_{t:.3f}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path


    def create_ties_config(base_model_path: str, expert_model_path: str, weight: float, 
                        density: float, output_path: str) -> str:
        """
        Create YAML config for TIES merge
        
        Args:
            base_model_path: Path to base model
            expert_model_path: Path to expert model
            weight: Weight for task vector (your λ)
            density: Fraction of parameters to keep
            output_path: Where to save merged model
            
        Returns:
            Path to YAML config file
        """
        
        weight = float(weight)  # ← ADD THIS LINE
        density = float(density)  # ← ADD THIS LINE
        
        config = {
            'merge_method': 'ties',
            'base_model': base_model_path,
            'models': [
                {
                    'model': expert_model_path,
                    'parameters': {
                        'weight': weight,
                        'density': density
                    }
                }
            ],
            'parameters': {
                'normalize': False,
                'int8_mask': False
            },
            'dtype': 'float32'
        }
        
        config_path = os.path.join(tempfile.gettempdir(), f'ties_config_{weight:.3f}_{density:.2f}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path


    def create_della_config(base_model_path: str, expert_model_path: str, weight: float,
                            density: float, epsilon: float, rescale: bool, output_path: str) -> str:
        """
        Create YAML config for DELLA merge
        
        Args:
            base_model_path: Path to base model
            expert_model_path: Path to expert model
            weight: Weight for task vector (your λ)
            density: Fraction of parameters to keep
            epsilon: Numerical stability constant
            rescale: Whether to rescale after pruning
            output_path: Where to save merged model
            
        Returns:
            Path to YAML config file
        """
        
        weight = float(weight)  # ← ADD THIS LINE
        density = float(density)  # ← ADD THIS LINE
        epsilon = float(epsilon)  # ← ADD THIS LINE
        
        config = {
            'merge_method': 'della',
            'base_model': base_model_path,
            'models': [
                {
                    'model': expert_model_path,
                    'parameters': {
                        'weight': weight,
                        'density': density,
                        'epsilon': epsilon
                    }
                }
            ],
            'parameters': {
                'rescale': rescale
            },
            'dtype': 'float32'
        }
        
        config_path = os.path.join(tempfile.gettempdir(), f'della_config_{weight:.3f}_{density:.2f}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path


    def merge_with_mergekit(
        base_model_path: str,
        expert_model_path: str,
        method: str,
        lambda_k: float,
        **kwargs):
        """
        Merge models using mergekit's actual implementation
        
        Args:
            base_model_path: Path to base model (can be HF model or local path)
            expert_model_path: Path to expert model
            method: 'slerp', 'ties', or 'della'
            lambda_k: Interpolation parameter [0, 1]
            output_dir: Temporary directory for merged model
            **kwargs: Method-specific parameters
            
        Returns:
            Merged state dict
        """

        print(f"\n{'='*70}")
        print(f"Creating Pseudo-Expert with Mergekit {method.upper()}")
        print(f"λ = {lambda_k:.3f}")
        print(f"{'='*70}")
        
        if not MERGEKIT_AVAILABLE:
            raise ImportError("Mergekit not available. Cannot perform merge.")
        
        # Create temp output directory with unique ID to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        temp_output = os.path.join(tempfile.gettempdir(), f'merge_{method}_{lambda_k:.3f}_{unique_id}')
        os.makedirs(temp_output, exist_ok=True)
        print("made the directory")
        
        config_path = None
        
        try:
            # Create appropriate config
            if method == 'slerp':
                config_path = create_slerp_config(
                    base_model_path, expert_model_path, lambda_k, temp_output
                )
            
            elif method == 'ties':
                density = kwargs.get('density', 0.9)
                config_path = create_ties_config(
                    base_model_path, expert_model_path, lambda_k, density, temp_output
                )
            
            elif method == 'della':
                density = kwargs.get('density', 0.9)
                epsilon = kwargs.get('epsilon', 1e-8)
                rescale = kwargs.get('rescale', True)
                config_path = create_della_config(
                    base_model_path, expert_model_path, lambda_k, 
                    density, epsilon, rescale, temp_output
                )
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            print(f"  Config created: {config_path}")
            
            # Load YAML as dictionary FIRST
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            print(f"  Config dict loaded, keys: {list(config_dict.keys())}")
            
            # Try multiple methods to create MergeConfiguration
            merge_config = None
            last_error = None
            
            # Method 1: Direct instantiation with **kwargs
            merge_config = MergeConfiguration(**config_dict)
            print(f"  ✓ MergeConfiguration created via direct instantiation")
            
            if merge_config is None:
                raise Exception(f"Failed to create MergeConfiguration. Last error: {last_error}")
            
            # Create merge options
            merge_options = MergeOptions(
                copy_tokenizer=False,
                lazy_unpickle=False,
                low_cpu_memory=False,
                write_model_card=False
            )
            print(f"  ✓ MergeOptions created")


            # Run merge
            print(f"  Running mergekit {method.upper()} merge...")

            if merge_options:
                run_merge(merge_config, temp_output, merge_options)
            else:
                run_merge(merge_config, temp_output)
            
            print(f"  ✓ Merge completed")
            
            # Load merged model state dict
            print(f"  Loading merged model from: {temp_output}")
            merged_model = BertForSequenceClassification.from_pretrained(temp_output,num_labels=self.num_labels).to(self.device)
            
            return merged_model 
        
        
        # except ValueError as e: 
        #     if "Circular reference detected" in str(e):
        #         print(f"  ⚠ Ignoring Pydantic serialization error")
                
        #     # merged_model = BertForSequenceClassification.from_pretrained(temp_output,num_labels=config.num_labels).to(config.device)
        #     # return merged_model 
                
        except Exception as e:
            print(f"  ✗ Merge failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_output):
                try:
                    shutil.rmtree(temp_output)
                    print(f"  ✓ Cleaned up temp directory")
                except Exception as e:
                    print(f"  ⚠ Could not clean up temp directory: {e}")



    def quadratic_interpolation_weight(lambda_val, curve_param=0.3):
        """
        Convert lambda to quadratic interpolation weight
        
        Args:
            lambda_val: Original lambda [0,1]
            curve_param: Curvature (0=linear, >0=convex, <0=concave)
        
        Returns:
            Quadratic weight for interpolation
        """
        # Quadratic function: w(λ) = aλ² + bλ + c
        # Constraints: w(0)=0, w(1)=1
        # This gives: w(λ) = curve_param*λ² + (1-curve_param)*λ
        return curve_param * lambda_val**2 + (1 - curve_param) * lambda_val

    # λ (lambda) is the global scale on whatever “important changes” the method kept.



    def get_interpolated_model(lambda_k,interpolation_name,theta_base,theta_exp):
        print(f"\n{'-'*70}")
        print(f"Interpolated Model (λ={lambda_k:.2f})")
        print(f"{'-'*70}")
        
        # Create interpolated model: θ_k = (1 - λ_k) * θ_base + λ_k * θ_exp
        # INTERPOLATE ALL PARAMETERS
        theta_k_model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels).to(self.device)
        
        # ToDo: we need to chnage only the classification layer/last layer weights. 
        # What does the .named_parameters do and is there a better way to get the last layer weigths
        
        # Ans: we are not getting the last layer weights. By solving the prop equation, its importnant that we 
        # itnerpoalte all the parameters. 
        
        
        if(interpolation_name=='linear'):
            with torch.no_grad():
                for name, param in theta_k_model.named_parameters():
                    param.copy_((1 - lambda_k) * theta_base[name] + lambda_k * theta_exp[name])
            return theta_k_model
        
        elif(interpolation_name=='quadratic'):
            quad_interpolation = quadratic_interpolation_weight(lambda_k, curve_param=0.3)
            with torch.no_grad():
                for name, param in theta_k_model.named_parameters():
                    param.copy_((1 - quad_interpolation) * theta_base[name] + quad_interpolation * theta_exp[name])
            return theta_k_model
        
        else: 
            return merge_with_mergekit(base_model_path,expert_model_path,interpolation_name,lambda_k)

