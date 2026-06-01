import os
from datasets import load_dataset
from .constants import DEFAULT_DATASET_ID, DEFAULT_DATASET_DIR

def download_default_dataset():
    """
    Downloads the default dataset from Hugging Face and returns the local path.
    The dataset is saved in a directory structure compatible with the rest of the code.
    """
    # Use a cache directory for the dataset
    cache_dir = os.path.expanduser("~/.cache/equiface")
    local_path = os.path.join(cache_dir, DEFAULT_DATASET_DIR)
    
    if not os.path.exists(local_path):
        print(f"Downloading default dataset '{DEFAULT_DATASET_ID}' to {local_path}...")
        
        # Load the dataset
        # Note: User might need to be logged in via huggingface-cli login
        try:
            ds = load_dataset(DEFAULT_DATASET_ID)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure you are logged in using `huggingface-cli login` if this is a gated dataset.")
            raise

        # Save the dataset to the local path in the expected directory structure
        # Assuming the dataset has a 'train' split and columns like 'image' and 'label' or 'id'
        # We need to know the structure. If it's an ImageFolder, it's easier.
        
        # Since we don't know the exact structure without being able to download it,
        # we'll assume it's a standard dataset and we'll save images into folders based on their identity.
        
        os.makedirs(local_path, exist_ok=True)
        
        # Iterate through all splits (usually 'train')
        for split in ds.keys():
            for i, example in enumerate(ds[split]):
                # Determine identity (folder name)
                # We'll check for common column names: 'label', 'identity', 'id', 'person'
                identity = None
                for col in ['identity', 'label', 'id', 'person', 'name', 'group', 'class']:
                    if col in example:
                        identity = str(example[col])
                        break
                
                if identity is None:
                    # Try to use a folder structure if available in file_name or similar
                    if 'file_name' in example and '/' in example['file_name']:
                        identity = os.path.dirname(example['file_name'])
                    else:
                        identity = "unknown"
                
                identity_dir = os.path.join(local_path, identity)
                os.makedirs(identity_dir, exist_ok=True)
                
                # Save image
                image = example['image']
                # Determine image name
                image_name = f"img_{i}.png" # Default to png
                if 'file_name' in example:
                    image_name = example['file_name']
                elif 'image_path' in example:
                    image_name = os.path.basename(example['image_path'])
                
                image.save(os.path.join(identity_dir, image_name))
        
        print("Download and extraction complete.")
    
    return local_path
