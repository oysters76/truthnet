import csv
import json
from pathlib import Path


def load_data(path, output_size=1):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = p.suffix.lower()
    if ext == ".json":
        return _load_json(p)
    elif ext == ".csv":
        return _load_csv(p, output_size)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .csv or .json")


def _load_json(path):
    with open(path) as f:
        raw = json.load(f)

    data = []
    for entry in raw:
        inputs, output = entry
        if isinstance(inputs, (int, float)):
            inputs = [inputs]
        data.append((inputs, output))

    return data


def _load_csv(path, output_size):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    try:
        float(rows[0][0])
        has_header = False
    except ValueError:
        has_header = True

    start = 1 if has_header else 0
    data = []
    for row in rows[start:]:
        values = [float(v) for v in row]
        inputs = values[:-output_size]
        if output_size == 1:
            output = values[-1]
        else:
            output = values[-output_size:]
        data.append((inputs, output))

    return data


def infer_sizes(data):
    inputs, output = data[0]
    input_size = len(inputs)
    output_size = 1 if isinstance(output, (int, float)) else len(output)
    return input_size, output_size
