import os
import random
import itertools
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import tensorflow.lite as tflite
from functools import lru_cache

from .image_utils import preprocess_image, get_embedding
from .logging_utils import log_results
from .constants import SUPPORTED_EXTENSIONS, DEFAULT_THRESHOLD, IMAGE_SIZE

def normalise(embedding):
    embedding = np.ravel(embedding)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Zero-norm embedding cannot be normalised.")
    return embedding / norm

@lru_cache(maxsize=5)
def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def verify(model_path, img1_path, img2_path, threshold=DEFAULT_THRESHOLD,
           image_size=IMAGE_SIZE):
    """Compares two images using a TFLite model and returns verification result."""
    interpreter = load_interpreter(model_path)
    interpreter.allocate_tensors()

    img1 = preprocess_image(img1_path, image_size)
    img2 = preprocess_image(img2_path, image_size)
    if img1 is None or img2 is None:
        return None

    emb1 = normalise(get_embedding(interpreter, img1))
    emb2 = normalise(get_embedding(interpreter, img2))
    similarity = np.dot(emb1, emb2)

    return similarity > threshold, similarity

def verify_pair(args):
    return verify(*args)

def process_pairs(image_pairs,
                  model_path,
                  use_multiprocessing=False,
                  num_cores=None,
                  threshold=DEFAULT_THRESHOLD,
                  image_size=IMAGE_SIZE):
    valid_results = []

    args_list = [(model_path, img1, img2, threshold, image_size) for img1, img2 in image_pairs]

    if use_multiprocessing:
        if num_cores is None or num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")

        with Pool(num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(verify_pair, args_list),
                                total=len(image_pairs), desc="Processing pairs", unit="pair"))
        valid_results = [r for r in results if r is not None]
    else:
        for args in tqdm(args_list, desc="Processing pairs", unit="pair"):
            result = verify_pair(args)
            if result is not None:
                valid_results.append(result)

    return valid_results

def FPR(dataset_dir,
        model_path,
        percentage=100,
        use_multiprocessing=False,
        num_cores=None,
        threshold=DEFAULT_THRESHOLD,
        image_size=IMAGE_SIZE):
    
    """
    Calculate False Positive Rates (FPRs) for the given TFlite model. 
    
    Args:
        dataset_dir (str): Path to demographic subfolder in main dataset directory
        model_path (str): Path to TFlite model
        Percentage (int): Percentage of total comparisons possible, i.e. if there are 64 possible comparisons
        percentage=50 will compute 32 comparisons.
        use_multiprocessing (bool): use multiprocessing to utilise multiple CPU threads
        num_cores (int): Number of CPU cores to be utilised
        threshold (float): Threshold for cosine similarity 
        image_size ((int,int)): Input dimension
    
    Returns:
        FPR_value: The false positive rate for the IDs in the given dataset_dir
    """
    
    subfolders = sorted([
        f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))
    ])

    image_pairs = [
        (os.path.join(dataset_dir, f1, img1), os.path.join(dataset_dir, f2, img2))
        for f1, f2 in itertools.combinations(subfolders, 2)
        for img1 in os.listdir(os.path.join(dataset_dir, f1))
        for img2 in os.listdir(os.path.join(dataset_dir, f2))
        if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS and
           os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    total_pairs = len(image_pairs)
    num_selected = int((percentage / 100) * total_pairs)
    image_pairs = random.sample(image_pairs, num_selected)

    results = process_pairs(image_pairs,
                            model_path,
                            use_multiprocessing,
                            num_cores,
                            threshold,
                            image_size)
    num_processed = len(results)

    FP = sum(match for match, _ in results)
    sims = [sim for _, sim in results]
    avg_similarity = sum(sims) / len(sims) if sims else 0

    FPR_value = FP / num_processed if num_processed > 0 else 0

    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FPR: {FPR_value:.4%}')
    print(f'Average similarity: {avg_similarity:.4f}')

    log_results(dataset_dir,
                model_path,
                "FPR",
                FPR_value,
                total_pairs,
                num_selected,
                FP=FP,
                mean_similarity=avg_similarity)

    return FPR_value

def FNR(dataset_dir,
        model_path,
        percentage=100,
        use_multiprocessing=False,
        num_cores=None,
        threshold=DEFAULT_THRESHOLD,
        image_size=IMAGE_SIZE):
    subfolders = sorted([
        f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))
    ])
    
    """
    Calculate False Negative Rates (FNRs) for the given TFlite model. 
    
    Args:
        dataset_dir (str): Path to demographic subfolder in main dataset directory
        model_path (str): Path to TFlite model
        Percentage (int): Percentage of total comparisons possible, i.e. if there are 64 possible comparisons
        percentage=50 will compute 32 comparisons.
        use_multiprocessing (bool): use multiprocessing to utilise multiple CPU threads
        num_cores (int): Number of CPU cores to be utilised
        threshold (float): Threshold for cosine similarity 
        image_size ((int,int)): Input dimension
    
    Returns:
        FNR_value: The false negative rate for the IDs in the given dataset_dir
    """
    image_pairs = [
        (os.path.join(dataset_dir, folder, img1), os.path.join(dataset_dir, folder, img2))
        for folder in subfolders
        for img1, img2 in itertools.combinations(
            sorted(os.listdir(os.path.join(dataset_dir, folder))), 2
        )
        if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS and
           os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    total_pairs = len(image_pairs)
    num_selected = int((percentage / 100) * total_pairs)
    image_pairs = random.sample(image_pairs, num_selected)

    results = process_pairs(image_pairs,
                            model_path,
                            use_multiprocessing,
                            num_cores,
                            threshold,
                            image_size)
    num_processed = len(results)

    FN = sum(not match for match, _ in results)
    sims = [sim for _, sim in results]
    avg_similarity = sum(sims) / len(sims) if sims else 0

    FNR_value = FN / num_processed if num_processed > 0 else 0

    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FNR: {FNR_value:.4%}')
    print(f'Average similarity: {avg_similarity:.4f}')

    log_results(dataset_dir,
                model_path,
                "FNR",
                FNR_value,
                total_pairs,
                num_selected,
                FN=FN,
                mean_similarity=avg_similarity)

    return FNR_value
