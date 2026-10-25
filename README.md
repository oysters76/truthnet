# TruthNet

A neural network built from scratch with NumPy. Train and run a 2-hidden-layer feedforward network from the command line.

## Install

```bash
uv sync
```

## Usage

### Train a model

```bash
truthnet train --data train.csv --epochs 500 --lr 0.01 --save model.npz
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Training data file (`.csv` or `.json`) |
| `--hidden` | `8` | Hidden layer size |
| `--output` | `1` | Output size |
| `--epochs` | `500` | Number of training epochs |
| `--lr` | `0.01` | Learning rate |
| `--test` | — | Optional test data file |
| `--save` | `model.npz` | Path to save trained model |

### Run predictions

Single input:

```bash
truthnet predict --model model.npz --input "1,0,1"
```

From a file:

```bash
truthnet predict --model model.npz --data test.csv
```

### Inspect a model

```bash
truthnet info --model model.npz
```

```
Model: model.npz
Architecture: 3 → 8 → 8 → 1
Total parameters: 113
  W1: (3, 8), b1: (1, 8)
  W2: (8, 8), b2: (1, 8)
  W3: (8, 1), b3: (1, 1)
```

## Data Formats

### CSV

Last column(s) are the output. Header row is optional (auto-detected).

```csv
x1,x2,x3,y
1,0,0,1
0,1,0,0
1,0,1,1
```

### JSON

Array of `[inputs, output]` pairs:

```json
[
  [[1, 0, 0], 1],
  [[0, 1, 0], 0],
  [[1, 0, 1], 1]
]
```

## Architecture

The network has two hidden layers with ReLU activation and a linear output:

```
input → W1 + b1 → ReLU → W2 + b2 → ReLU → W3 + b3 → output
```

Training uses MSE loss with backpropagation and gradient descent.

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

### Project structure

```
src/truthnet/
├── __init__.py      # Package exports
├── __main__.py      # python -m truthnet entry point
├── cli.py           # Argparse CLI (train, predict, info)
├── data.py          # CSV/JSON data loading
└── network.py       # Network class (feedforward, backprop, save/load)
tests/
├── test_cli.py      # CLI integration tests
├── test_data.py     # Data loading unit tests
└── test_network.py  # Network unit tests
```
