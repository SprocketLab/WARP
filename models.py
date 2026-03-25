
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
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions
    MERGEKIT_AVAILABLE = True
    print("✓ Mergekit imported successfully")
    
except ImportError as e:
    MERGEKIT_AVAILABLE = False
    print(f"⚠ Mergekit not available: {e}")
    
    
def get_interpolated_model(self,lambda_k,interpolation_name,theta_base,theta_exp):
    """
    Create an interpolated pseudo-expert model between base and expert models.
    
    This is the main entry point for generating pseudo-expert models. It supports
    multiple interpolation methods and delegates to the appropriate implementation:
    
    - 'linear': Direct parameter averaging
    - 'quadratic': Non-linear parameter averaging with curvature
    - 'slerp', 'ties', 'della': Advanced merging via mergekit
    
    All parameters are interpolated (not just the last layer) to maintain
    model coherence and ensure the interpolated model represents a valid
    point on the fine-tuning trajectory.
    
    Args:
        lambda_k (float): Interpolation coefficient [0, 1]
            - λ=0: Returns model equivalent to base model
            - λ=1: Returns model equivalent to expert model
            - 0<λ<1: Returns interpolated pseudo-expert
        interpolation_name (str): Method name ('linear', 'quadratic', 'slerp', 
                                    'ties', 'della')
        theta_base (dict): Base model parameters {name: tensor}
        theta_exp (dict): Expert model parameters {name: tensor}
        
    Returns:
        BertForSequenceClassification: Interpolated model on self.device
        
    Note:
        - Creates a fresh model from model_name, then copies interpolated parameters
        - torch.no_grad() context ensures no gradient tracking during interpolation
        - For mergekit methods, delegates to merge_with_mergekit()
        
    Implementation Detail:
        Linear: θ_k = (1-λ)·θ_base + λ·θ_exp
        Quadratic: θ_k = (1-w(λ))·θ_base + w(λ)·θ_exp where w is non-linear
    """
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
    # itnerpoalte all the parameters.tion
    
    

class Model:
    """
    Handles model interpolation and merging for creating pseudo-expert models.
    
    This class provides multiple methods for interpolating between a base model
    and an expert model to create intermediate pseudo-expert models. It supports:
    
    1. Linear interpolation: θ_k = (1-λ)θ_base + λ·θ_exp
    2. Quadratic interpolation: Non-linear weighting with curvature control
    3. SLERP (Spherical Linear Interpolation): Smooth interpolation on sphere
    4. TIES (Task Intersection with Expert Selection): Parameter pruning + merging
    5. DELLA: Advanced merging with density-based pruning
    
    The class integrates with the mergekit library for advanced merging methods
    and handles all the configuration, execution, and model loading.
    
    Attributes:
        tokenizer: HuggingFace tokenizer for the model
        base_model_path (str): Path to saved base model
        expert_model_path (str): Path to saved expert model
        no_of_pseudoexperts (int): Number of interpolated models to create
        device (torch.device): Device for model operations (CPU or CUDA)
        model_name (str): HuggingFace model identifier (e.g., 'bert-base-uncased')
        num_labels (int): Number of classification labels
    """
    
    def __init__(self,tokenizer,base_model_path,expert_model_path,no_of_pseudoexperts,device,model_name,num_labels):
        self.tokenizer = tokenizer
        self.base_model_path = base_model_path 
        self.expert_model_path = expert_model_path
        self.no_of_pseudoexperts = no_of_pseudoexperts
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels


    def _create_slerp_config(self,t: float, output_path: str) -> str:
        """
        Create YAML configuration file for SLERP (Spherical Linear Interpolation) merge.
        
        SLERP interpolates model parameters along the surface of a hypersphere,
        providing smooth interpolation that preserves parameter magnitude better
        than linear interpolation.
        
        Args:
            t (float): Interpolation parameter in [0, 1]
                      t=0 gives base model, t=1 gives expert model
            output_path (str): Directory where merged model will be saved
            
        Returns:
            str: Path to the generated YAML configuration file
            
        Note:
            - Uses layer_range [0, -1] to merge all layers
            - dtype='float32' for numerical stability
            - Config saved to temp directory with unique filename
        """
        
        t = float(t) 
        config = {
            'merge_method': 'slerp',
            'base_model': self.base_model_path,
            'slices': [
                {
                    'sources': [
                        {'model': self.base_model_path, 'layer_range': [0, -1]},
                        {'model': self.expert_model_path, 'layer_range': [0, -1]}
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


    def _create_ties_config(self,weight: float, density: float, output_path: str) -> str:
        """
        Create YAML configuration file for TIES (Task Intersection with Expert Selection) merge.
        
        TIES merging:
        1. Computes task vector: τ = θ_expert - θ_base
        2. Prunes parameters based on magnitude (keeps top 'density' fraction)
        3. Resolves conflicts by selecting parameters with consistent signs
        4. Scales by 'weight': θ_merged = θ_base + weight * τ_pruned
        
        This method is effective for merging task-specific adaptations while
        avoiding parameter interference.
        
        Args:
            weight (float): Scaling factor for the task vector (equivalent to λ)
                          Higher values move closer to expert model
            density (float): Fraction of parameters to keep after pruning [0, 1]
                           0.9 means keep top 90% of parameters by magnitude
            output_path (str): Directory where merged model will be saved
            
        Returns:
            str: Path to the generated YAML configuration file
            
        Note:
            - normalize=False: Don't normalize task vectors
            - int8_mask=False: Use float masks for precision
        """
        
        weight = float(weight)  # ← ADD THIS LINE
        density = float(density)  # ← ADD THIS LINE
        
        config = {
            'merge_method': 'ties',
            'base_model': self.base_model_path,
            'models': [
                {
                    'model': self.expert_model_path,
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


    def _create_della_config(self,weight: float,density: float, epsilon: float, rescale: bool, output_path: str) -> str:
        """
        Create YAML configuration file for DELLA (Density-based Expert Layer Aggregation) merge.
        
        DELLA is an advanced merging method that:
        1. Computes task vectors like TIES
        2. Uses density-based parameter selection with numerical stability (epsilon)
        3. Optionally rescales merged parameters to maintain model capacity
        
        This method often outperforms TIES by better handling parameter magnitudes
        and providing more stable merging.
        
        Args:
            weight (float): Scaling factor for the task vector (equivalent to λ)
            density (float): Fraction of parameters to keep [0, 1]
            epsilon (float): Numerical stability constant (default: 1e-8)
                           Prevents division by zero in density computation
            rescale (bool): Whether to rescale parameters after merging
                          True recommended for maintaining model performance
            output_path (str): Directory where merged model will be saved
            
        Returns:
            str: Path to the generated YAML configuration file
            
        Note:
            - More sophisticated than TIES with better numerical stability
            - Rescaling helps preserve model capacity after pruning
        """
        
        weight = float(weight)  # ← ADD THIS LINE
        density = float(density)  # ← ADD THIS LINE
        epsilon = float(epsilon)  # ← ADD THIS LINE
        
        config = {
            'merge_method': 'della',
            'base_model': self.base_model_path,
            'models': [
                {
                    'model': self.expert_model_path,
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


    def merge_with_mergekit(self,method: str,lambda_k: float,**kwargs):
        """
        Merge base and expert models using mergekit's implementation.
        
        This is the main method for creating pseudo-expert models using advanced
        merging techniques (SLERP, TIES, DELLA). It:
        1. Creates appropriate YAML configuration for the method
        2. Initializes mergekit's MergeConfiguration and MergeOptions
        3. Executes the merge operation
        4. Loads the merged model and returns it
        5. Cleans up temporary files
        
        The merging happens in a temporary directory with a unique ID to avoid
        conflicts when running multiple merges in parallel.
        
        Args:
            method (str): Merging method - 'slerp', 'ties', or 'della'
            lambda_k (float): Interpolation parameter [0, 1]
                            Controls how much to blend base vs expert
            **kwargs: Method-specific parameters:
                - For TIES: density (default: 0.9)
                - For DELLA: density (default: 0.9), epsilon (default: 1e-8), 
                           rescale (default: True)
            
        Returns:
            BertForSequenceClassification: Merged model loaded on self.device
            
        Raises:
            ImportError: If mergekit is not available
            ValueError: If method is not recognized
            Exception: If merge operation fails
            
        Side Effects:
            - Creates temporary directory for merge output
            - Saves YAML config to temp directory
            - Cleans up temp directory after loading model
            - Prints detailed progress information
            
        Note:
            - Requires mergekit library to be installed
            - Uses unique UUIDs to allow parallel execution
            - Tokenizer is not copied (copy_tokenizer=False)
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
                config_path = self._create_slerp_config(lambda_k, temp_output)
            
            elif method == 'ties':
                density = kwargs.get('density', 0.9)
                config_path = self._create_ties_config(lambda_k, density, temp_output)
            
            elif method == 'della':
                density = kwargs.get('density', 0.9)
                epsilon = kwargs.get('epsilon', 1e-8)
                rescale = kwargs.get('rescale', True)
                config_path = self._create_della_config(lambda_k, density, epsilon, rescale, temp_output)
            
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



    def quadratic_interpolation_weight(self,lambda_val, curve_param=0.3):
        """
        Convert linear lambda to quadratic interpolation weight.
        
        This creates a non-linear interpolation trajectory that can be convex
        or concave depending on the curve_param. The function satisfies:
        - w(0) = 0 (returns base model)
        - w(1) = 1 (returns expert model)
        
        Formula: w(λ) = curve_param·λ² + (1-curve_param)·λ
        
        Args:
            lambda_val (float): Linear interpolation parameter [0, 1]
            curve_param (float): Curvature parameter (default: 0.3)
                - curve_param = 0: Linear interpolation (w = λ)
                - curve_param > 0: Convex curve (slow start, fast finish)
                - curve_param < 0: Concave curve (fast start, slow finish)
        
        Returns:
            float: Quadratic interpolation weight [0, 1]
            
        Example:
            >>> quadratic_interpolation_weight(0.5, 0.3)
            0.425  # Less than 0.5, slower progress initially
        """
        # Quadratic function: w(λ) = aλ² + bλ + c
        # Constraints: w(0)=0, w(1)=1
        # This gives: w(λ) = curve_param*λ² + (1-curve_param)*λ
        return curve_param * lambda_val**2 + (1 - curve_param) * lambda_val

    # λ (lambda) is the global scale on whatever “important changes” the method kept.



    def get_interpolated_model(self,lambda_k,interpolation_name,theta_base,theta_exp):
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
            quad_interpolation = self.quadratic_interpolation_weight(lambda_k, curve_param=0.3)
            with torch.no_grad():
                for name, param in theta_k_model.named_parameters():
                    param.copy_((1 - quad_interpolation) * theta_base[name] + quad_interpolation * theta_exp[name])
            return theta_k_model
        
        else: 
            return self.merge_with_mergekit(interpolation_name,lambda_k)

