import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import random

class DatasetCollector:
    """Collects activations from a transformer model for a given dataset."""
    
    def __init__(self, model_name: str = "gpt2-small", device: str = None):
        """
        Initialize the DatasetCollector.
        
        Args:
            model_name: Name of the pretrained model to load
            device: Device to run on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HookedTransformer.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    @staticmethod
    def store_acts(tensor, hook):
        """Hook function to store activations."""
        # tensor: [batch, seq_len, d_model]
        hook.ctx["acts"] = tensor[:, -1, :].detach().cpu()
    
    def collect(self, corpus: list[tuple[str, int]], output_prefix: str = "acts_layer", batch_size: int = 8, test_split: float = 0.2, random_seed: int = 42):
        """
        Collect activations for the given corpus in batches.
        
        Args:
            corpus: List of (text, label) tuples
            output_prefix: Prefix for output files
            batch_size: Number of texts to process in each batch
            test_split: Fraction of data to save as test split (default: 0.2)
            random_seed: Random seed for reproducible train/test split (default: 42)
        """
        # Initialize storage for each layer - list of individual examples
        all_layer_data = {i: [] for i in range(len(self.model.blocks))}
        
        # Process corpus in batches
        for batch_start in tqdm(range(0, len(corpus), batch_size)):
            batch_end = min(batch_start + batch_size, len(corpus))
            batch = corpus[batch_start:batch_end]
            
            texts = [text for text, _ in batch]
            labels = [label for _, label in batch]
            
            # Tokenize batch
            tokens = self.model.to_tokens(texts).to(self.device)

            # Run with hook
            _ = self.model.run_with_hooks(
                tokens,
                fwd_hooks=[
                    (f"blocks.{i}.mlp.hook_post", self.store_acts) 
                    for i in range(len(self.model.blocks))
                ]
            )

            # Collect activations for each layer
            for i in range(len(self.model.blocks)):
                acts = self.model.hook_dict[f"blocks.{i}.mlp.hook_post"].ctx["acts"]
                
                # Store each example separately
                for j in range(len(texts)):
                    example = {
                        "acts": acts[j:j+1],  # Keep dimension [1, d_model]
                        "text": texts[j],
                        "label": labels[j]
                    }
                    all_layer_data[i].append(example)
        
        # Shuffle and split data into train/test for each layer
        random.seed(random_seed)
        for i in range(len(self.model.blocks)):
            # Shuffle the data
            data = all_layer_data[i]
            indices = list(range(len(data)))
            random.shuffle(indices)
            
            # Calculate split point
            test_size = int(len(data) * test_split)
            
            # Split into train and test
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            train_data = [data[idx] for idx in train_indices]
            test_data = [data[idx] for idx in test_indices]
            
            # Save both splits
            torch.save(train_data, f"{output_prefix}_{i}_train.pt")
            torch.save(test_data, f"{output_prefix}_{i}_test.pt")
