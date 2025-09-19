import json
import numpy as np
from pathlib import Path

def compute_gveval_score(jsonl_path):
    """Compute mean overall score from a JSONL results file."""
    overall_scores = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                overall_scores.append(data["result"]["overall"])
            except Exception as e:
                print(f"Skipping line in {jsonl_path}: {e}")
    if not overall_scores:
        return None
    return float(np.mean(overall_scores))


def process_files(
    file_paths,
    output_json=r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\gveval_scores.json"
):
    """Compute scores for multiple JSONL files and save results into a structured JSON."""
    results = {"drawbench": {}, "mscoco": {}}

    for path in file_paths:
        score = compute_gveval_score(path)
        path_obj = Path(path)

        # Detect dataset (drawbench or mscoco) based on parent folder
        if "drawbench_gveval" in str(path_obj):
            dataset = "drawbench"
        elif "mscoco_gveval" in str(path_obj):
            dataset = "mscoco"
        else:
            dataset = "other"

        # Detect model (fluxdev, sd2, sdxl) from folder name
        model = path_obj.parent.name  

        if dataset not in results:
            results[dataset] = {}
        results[dataset][model] = score

        print(f"{dataset} | {model}: {score}")

    # Save results into JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to {output_json}")


# --- Example usage ---
files = [
    r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\drawbench_gveval\fluxdev\results_fluxdev_gveval.jsonl",
    r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\drawbench_gveval\sd2\results_sd2_gveval.jsonl",
    r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\drawbench_gveval\sdxl\results_sdxl_gveval.jsonl",
    r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\mscoco_gveval\fluxdev\mscoco_fluxdev_gveval.jsonl",
    r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\mscoco_gveval\sd2\mscoco_sd2_gveval.jsonl",
    r"C:\Users\roopa\OneDrive\Desktop\Gveval_metric\mscoco_gveval\sdxl\mscoco_sdxl_gveval.jsonl"
]

process_files(files)
