import sys
import os
import csv
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python aggregate_docking_results.py <score_files...> <output_csv>")
        sys.exit(1)

    output_csv = sys.argv[-1]

    input_files = sys.argv[1:-1]

    results = []

    print(f"Aggregating results from {len(input_files)} potential files...")

    for file_path in input_files:
        path = Path(file_path)

        if not path.name.endswith(".score.txt"):
            continue

        if not path.exists() or path.stat().st_size == 0:
            continue

        try:
            with open(path, 'r') as f:
                content = f.read().strip()
                if not content:
                    continue

                score = float(content)
                model_name = path.name.replace(".score.txt", "")
                candidate_name = path.parent.name

                results.append({
                    "candidate": candidate_name,
                    "model": model_name,
                    "docking_score": score
                })
        except ValueError:
            print(f"Warning: Could not parse score from {path}. Skipping.")
        except Exception as e:
            print(f"Error reading {path}: {e}")

    results.sort(key=lambda x: x["docking_score"])

    print(f"Writing {len(results)} valid results to {output_csv}")

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['candidate', 'model', 'docking_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
