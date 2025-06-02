import os
import json
import numpy as np
from datetime import datetime


def make_json_safe(obj):
    """Rekurencyjna konwersja obiektów do postaci JSON-serializowalnej."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.ndarray, list, tuple)):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (np.generic, np.number)):
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # fallback – np. device, object, itp.


def log_experiment(result_dict, model_params, attack_type):
    raw_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type": attack_type,
        "model_params": model_params,
        **result_dict
    }

    # 🔒 Zabezpieczenie przed numpy.ndarray i innymi nieserializowalnymi obiektami
    log_entry = make_json_safe(raw_entry)

    os.makedirs("data/autoencoder/logs", exist_ok=True)
    log_file = "data/autoencoder/logs/experiments_log.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def save_run_snapshot(snapshot_dict, extra_meta=None):
    """Zapisuje pełen snapshot bieżącego przebiegu."""
    safe_snapshot = make_json_safe(snapshot_dict)

    if extra_meta:
        safe_snapshot["meta"] = make_json_safe(extra_meta)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/autoencoder/logs/log_{timestamp}.jsonl"

    os.makedirs("data/autoencoder/logs", exist_ok=True)

    with open(filename, "w") as f:
        f.write(json.dumps(safe_snapshot, indent=2))

    print(f"📁 Snapshot zapisany do: {filename}")
