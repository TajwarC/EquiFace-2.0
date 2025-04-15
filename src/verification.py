import os
import random
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import cosine
import tensorflow.lite as tflite

from .image_utils import preprocess_image, get_embedding
from .logging_utils import log_results
from .constants import SUPPORTED_EXTENSIONS, DEFAULT_THRESHOLD


def verify(model_path, img1_path, img2_path, threshold=DEFAULT_THRESHOLD):
    """Compares two images using a TFLite model and returns verification result and similarity score."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    img1, img2 = preprocess_image(img1_path), preprocess_image(img2_path)
    if img1 is None or img2 is None:
        return None

    emb1, emb2 = get_embedding(interpreter, img1), get_embedding(interpreter, img2)
    similarity = 1 - cosine(emb1, emb2)

    return similarity > threshold, similarity


def verify_pair(args):
    """Wrapper for parallel processing of image pairs."""
    img1, img2, model_path = args
    return verify(model_path, img1, img2)


def process_pairs(image_pairs, model_path, use_multiprocessing=False, num_cores=None):
    """Processes image pairs with optional multiprocessing."""
    valid_results = []

    if use_multiprocessing:
        if num_cores is None or num_cores < 1 or num_cores > cpu_count():
            raise ValueError(f"num_cores must be between 1 and {cpu_count()}")

        with Pool(num_cores) as pool:
            results = list(tqdm(
                pool.imap_unordered(verify_pair, [(pair[0], pair[1], model_path) for pair in image_pairs]),
                total=len(image_pairs), desc="Processing pairs", unit="pair"
            ))
        valid_results = [r for r in results if r is not None]
    else:
        for pair in tqdm(image_pairs, desc="Processing pairs", unit="pair"):
            result = verify_pair((pair[0], pair[1], model_path))
            if result is not None:
                valid_results.append(result)

    return valid_results


def FPR(dataset_dir, model_path, percentage=100, use_multiprocessing=False, num_cores=None):
    """Calculates the False Positive Rate (FPR) for a TFLite model."""
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

    results = process_pairs(image_pairs, model_path, use_multiprocessing, num_cores)
    num_processed = len(results)

    FP = sum(match for match, _ in results)
    sims = [sim for _, sim in results]
    avg_similarity = sum(sims) / len(sims) if sims else 0

    FPR_value = FP / num_processed if num_processed > 0 else 0

    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FPR: {FPR_value:.4%}')
    print(f'Average Distance: {avg_similarity:.4f}')
    
    log_results(dataset_dir, model_path, "FPR", FPR_value, total_pairs, num_processed, FP=FP,
                avg_distance=avg_similarity, mean_similarity=avg_similarity)

    return FPR_value


def FNR(dataset_dir, model_path, percentage=100, use_multiprocessing=False, num_cores=None):
    """Calculates the False Negative Rate (FNR) for a TFLite model."""
    subfolders = sorted([
        f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))
    ])

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

    results = process_pairs(image_pairs, model_path, use_multiprocessing, num_cores)
    num_processed = len(results)

    FN = sum(not match for match, _ in results)
    sims = [sim for _, sim in results]
    avg_similarity = sum(sims) / len(sims) if sims else 0

    FNR_value = FN / num_processed if num_processed > 0 else 0

    print(f'Total possible pairs: {total_pairs}')
    print(f'Processed pairs: {num_processed}')
    print(f'Mean FNR: {FNR_value:.4%}')
    print(f'Average Distance: {avg_similarity:.4f}')
    
    log_results(dataset_dir, model_path, "FNR", FNR_value, total_pairs, num_processed, FN=FN,
                avg_distance=avg_similarity, mean_similarity=avg_similarity)

    return FNR_value
