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
from .dataset_utils import download_default_dataset


def normalise(embedding):
    embedding = np.ravel(embedding)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Zero-norm embedding cannot be normalised.")
    return embedding / norm


@lru_cache(maxsize=5)
def load_interpreter(model_path):
    """Load and cache a TFLite interpreter along with pre-fetched tensor indices.

    The interpreter and its tensor indices are cached per model path to avoid
    repeated I/O and tensor-detail lookups on every inference call.

    Note: TFLite interpreters are not thread-safe, but ``use_multiprocessing``
    spawns separate *processes* (not threads), each with their own interpreter
    cache, so sharing via lru_cache is safe here.

    Returns:
        tuple: (interpreter, input_tensor_index, output_tensor_index)
    """
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    return interpreter, input_idx, output_idx


def verify(model_path, img1_path, img2_path, threshold=DEFAULT_THRESHOLD,
           image_size=IMAGE_SIZE):
    """Compares two images using a TFLite model and returns a verification result.

    Args:
        model_path (str): Path to the TFLite model file.
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        threshold (float): Cosine similarity threshold for a positive match.
        image_size (int): Square input dimension for the model.

    Returns:
        tuple[bool, float] or None: (is_match, similarity_score), or None if
        preprocessing fails (e.g. no face detected in either image).
    """
    interpreter, input_idx, output_idx = load_interpreter(model_path)

    img1 = preprocess_image(img1_path, image_size)
    img2 = preprocess_image(img2_path, image_size)
    if img1 is None or img2 is None:
        return None

    emb1 = normalise(get_embedding(interpreter, img1, input_idx, output_idx))
    emb2 = normalise(get_embedding(interpreter, img2, input_idx, output_idx))
    similarity = float(np.dot(emb1, emb2))

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
        max_cores = cpu_count()
        if num_cores is None or num_cores < 1 or num_cores > max_cores:
            raise ValueError(f"num_cores must be between 1 and {max_cores}.")

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


def _compute_verification_metric(
    metric,
    dataset_dir=None,
    model_path=None,
    percentage=100,
    use_multiprocessing=False,
    num_cores=None,
    threshold=DEFAULT_THRESHOLD,
    image_size=IMAGE_SIZE
):
    """Shared implementation for FPR and FNR computation.

    Args:
        metric (str): Either ``"FPR"`` (cross-identity / impostor pairs) or
                      ``"FNR"`` (within-identity / genuine pairs).
        dataset_dir (str, optional): Path to demographic subfolder in main dataset
                                     directory. If None, the default dataset is downloaded.
        model_path (str): Path to TFLite model.
        percentage (int): Percentage of total comparisons to run (1–100). For
                          example, with 64 possible pairs, percentage=50 runs 32.
        use_multiprocessing (bool): Use multiprocessing to utilise multiple CPU cores.
        num_cores (int): Number of CPU cores to utilise.
        threshold (float): Cosine similarity threshold for a positive match.
        image_size (int): Square input dimension for the model.

    Returns:
        float: The computed metric value (0.0–1.0).
    """
    if dataset_dir is None:
        dataset_dir = download_default_dataset()

    if model_path is None:
        raise ValueError("model_path must be provided.")

    if not (0 < percentage <= 100):
        raise ValueError("percentage must be between 1 and 100 (inclusive).")

    subfolders = sorted([
        f for f in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, f))
    ])

    if metric == "FPR":
        # Cross-identity pairs: images from *different* identity folders (impostor comparisons)
        image_pairs = [
            (os.path.join(dataset_dir, f1, img1), os.path.join(dataset_dir, f2, img2))
            for f1, f2 in itertools.combinations(subfolders, 2)
            for img1 in os.listdir(os.path.join(dataset_dir, f1))
            for img2 in os.listdir(os.path.join(dataset_dir, f2))
            if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS
            and os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS
            and os.path.isfile(os.path.join(dataset_dir, f1, img1))
            and os.path.isfile(os.path.join(dataset_dir, f2, img2))
        ]
    else:
        # Within-identity pairs: images from the *same* identity folder (genuine comparisons)
        image_pairs = [
            (os.path.join(dataset_dir, folder, img1), os.path.join(dataset_dir, folder, img2))
            for folder in subfolders
            for img1, img2 in itertools.combinations(
                sorted(os.listdir(os.path.join(dataset_dir, folder))), 2
            )
            if os.path.splitext(img1)[1].lower() in SUPPORTED_EXTENSIONS
            and os.path.splitext(img2)[1].lower() in SUPPORTED_EXTENSIONS
            and os.path.isfile(os.path.join(dataset_dir, folder, img1))
            and os.path.isfile(os.path.join(dataset_dir, folder, img2))
        ]

    total_pairs = len(image_pairs)
    num_selected = int((percentage / 100) * total_pairs)
    image_pairs = random.sample(image_pairs, num_selected)

    results = process_pairs(image_pairs, model_path, use_multiprocessing,
                            num_cores, threshold, image_size)
    num_processed = len(results)

    # Vectorised stats via NumPy — avoids repeated Python-level iteration
    matches = np.array([m for m, _ in results], dtype=bool)
    sims = np.array([s for _, s in results], dtype=np.float64)

    avg_similarity = float(sims.mean()) if num_processed > 0 else 0.0

    if metric == "FPR":
        count = int(matches.sum())                  # False Positives
        metric_value = count / num_processed if num_processed > 0 else 0.0
        log_kwargs = {"FP": count}
    else:
        count = int((~matches).sum())               # False Negatives
        metric_value = count / num_processed if num_processed > 0 else 0.0
        log_kwargs = {"FN": count}

    print(f"Total possible pairs: {total_pairs}")
    print(f"Processed pairs:      {num_processed}")
    print(f"Mean {metric}:         {metric_value:.4%}")
    print(f"Average similarity:   {avg_similarity:.4f}")

    log_results(
        dataset_dir, model_path, metric, metric_value,
        total_pairs, num_selected,
        mean_similarity=avg_similarity,
        **log_kwargs
    )

    return metric_value


def calculate_fpr(dataset_dir=None,
                  model_path=None,
                  percentage=100,
                  use_multiprocessing=False,
                  num_cores=None,
                  threshold=DEFAULT_THRESHOLD,
                  image_size=IMAGE_SIZE):
    """
    Calculate the False Positive Rate (FPR) for the given TFLite model.

    Compares images across *different* identities (impostor pairs). A false
    positive occurs when the model incorrectly matches two images of different people.

    Args:
        dataset_dir (str, optional): Path to demographic subfolder in main dataset
                                     directory. If None, the default dataset is downloaded.
        model_path (str): Path to TFLite model.
        percentage (int): Percentage of total comparisons to run (1 to 100). For
                          example, with 64 possible pairs, percentage=50 runs 32.
        use_multiprocessing (bool): Use multiprocessing to utilise multiple CPU cores.
        num_cores (int): Number of CPU cores to utilise.
        threshold (float): Cosine similarity threshold for a positive match.
        image_size (int): Square input dimension for the model.

    Returns:
        float: The false positive rate for the IDs in the given dataset_dir.
    """
    return _compute_verification_metric(
        "FPR", dataset_dir, model_path, percentage,
        use_multiprocessing, num_cores, threshold, image_size
    )


def calculate_fnr(dataset_dir=None,
                  model_path=None,
                  percentage=100,
                  use_multiprocessing=False,
                  num_cores=None,
                  threshold=DEFAULT_THRESHOLD,
                  image_size=IMAGE_SIZE):
    """
    Calculate the False Negative Rate (FNR) for the given TFLite model.

    Compares images within the *same* identity (genuine pairs). A false negative
    occurs when the model fails to match two images of the same person.

    Args:
        dataset_dir (str, optional): Path to demographic subfolder in main dataset
                                     directory. If None, the default dataset is downloaded.
        model_path (str): Path to TFLite model.
        percentage (int): Percentage of total comparisons to run (1–100). For
                          example, with 64 possible pairs, percentage=50 runs 32.
        use_multiprocessing (bool): Use multiprocessing to utilise multiple CPU cores.
        num_cores (int): Number of CPU cores to utilise.
        threshold (float): Cosine similarity threshold for a positive match.
        image_size (int): Square input dimension for the model.

    Returns:
        float: The false negative rate for the IDs in the given dataset_dir.
    """
    return _compute_verification_metric(
        "FNR", dataset_dir, model_path, percentage,
        use_multiprocessing, num_cores, threshold, image_size
    )

FPR = calculate_fpr
FNR = calculate_fnr
