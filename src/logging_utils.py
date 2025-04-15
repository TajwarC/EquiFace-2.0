import os
import yaml
from .constants import LOG_FILE

def log_results(
    dataset_dir,
    model_path,
    metric,
    value,
    total_pairs,
    num_selected,
    FP=None,
    FN=None,
    avg_distance=None,
    mean_similarity=None,
):
    """
    Logs evaluation results to a YAML file.
    
    Args:
        dataset_dir (str): Path to the dataset.
        model_path (str): Path to the TFLite model.
        metric (str): Either "FPR" or "FNR".
        value (float): Calculated value for the metric.
        total_pairs (int): Total number of possible pairs.
        num_selected (int): Number of pairs that were actually processed.
        FP (int, optional): False Positives count.
        FN (int, optional): False Negatives count.
        avg_distance (float, optional): Average similarity or distance score.
        mean_similarity (float, optional): Alias for avg_distance (used for logging clarity).
    """
    log_entry = {
        "dataset": dataset_dir,
        "model_name": model_path,
        "metric": metric,
        "value": float(round(value, 4)),
        "total_pairs": total_pairs,
        "num_selected": num_selected,
    }

    if FP is not None:
        log_entry["False Positives"] = int(FP)
    if FN is not None:
        log_entry["False Negatives"] = int(FN)
    if avg_distance is not None:
        log_entry["average_distance"] = float(round(avg_distance, 4))
    if mean_similarity is not None:
        log_entry["mean_similarity"] = float(round(mean_similarity, 4))

    # Load existing logs if available
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            try:
                logs = yaml.safe_load(file) or []
            except yaml.YAMLError:
                logs = []

    logs.append(log_entry)

    # Write updated logs
    with open(LOG_FILE, "w") as file:
        yaml.safe_dump(logs, file, default_flow_style=False)

    print(f"Logged {metric} result: {value:.4%} in {LOG_FILE}")

