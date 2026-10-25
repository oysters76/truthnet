import argparse
import sys

import numpy as np

from truthnet.data import infer_sizes, load_data
from truthnet.network import Network


def cmd_train(args):
    train_data = load_data(args.data, output_size=args.output)
    input_size, output_size = infer_sizes(train_data)

    if args.output != 1:
        output_size = args.output

    print(f"Training data: {len(train_data)} samples")
    print(f"Architecture: {input_size} → {args.hidden} → {args.hidden} → {output_size}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.lr}")
    print()

    net = Network(input_size, args.hidden, output_size)
    log_every = max(1, args.epochs // 10)
    net.optimize(
        train_data, learning_rate=args.lr, epochs=args.epochs, log_every=log_every
    )

    save_path = args.save
    net.save(save_path)
    print(f"\nModel saved to {save_path}")

    if args.test:
        test_data = load_data(args.test, output_size=output_size)
        print(f"\n--- Test Results ({len(test_data)} samples) ---")
        total_loss = 0
        for ix, y in test_data:
            y_hat = net.predict(np.array(ix))
            loss = net.L(y, y_hat).item()
            total_loss += loss
            predicted = y_hat[0][0] if output_size == 1 else y_hat[0].tolist()
            print(
                f"  Input: {ix}, Expected: {y}, Predicted: {predicted:.4f}, Loss: {loss:.6f}"
            )
        avg_loss = total_loss / len(test_data)
        print(f"  Average loss: {avg_loss:.6f}")


def cmd_predict(args):
    net = Network.load(args.model)

    if args.input:
        values = [float(v.strip()) for v in args.input.split(",")]
        x = np.array(values)
        y_hat = net.predict(x)
        if net.output_size == 1:
            print(f"Prediction: {y_hat[0][0]:.4f}")
        else:
            print(f"Prediction: {y_hat[0].tolist()}")

    elif args.data:
        data = load_data(args.data, output_size=net.output_size)
        print(f"Predictions ({len(data)} samples):")
        for ix, y in data:
            y_hat = net.predict(np.array(ix))
            predicted = y_hat[0][0] if net.output_size == 1 else y_hat[0].tolist()
            print(f"  Input: {ix}, Expected: {y}, Predicted: {predicted:.4f}")

    else:
        print("Error: provide --input or --data", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    net = Network.load(args.model)
    total_params = (
        net.W1.size
        + net.b1.size
        + net.W2.size
        + net.b2.size
        + net.W3.size
        + net.b3.size
    )
    print(f"Model: {args.model}")
    print(
        f"Architecture: {net.input_size} → {net.hidden_size} → {net.hidden_size} → {net.output_size}"
    )
    print(f"Total parameters: {total_params}")
    print(f"  W1: {net.W1.shape}, b1: {net.b1.shape}")
    print(f"  W2: {net.W2.shape}, b2: {net.b2.shape}")
    print(f"  W3: {net.W3.shape}, b3: {net.b3.shape}")


def main():
    parser = argparse.ArgumentParser(
        prog="truthnet",
        description="Train and run a 2-hidden-layer neural network from scratch",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--data", required=True, help="Training data file (.csv or .json)"
    )
    train_parser.add_argument(
        "--hidden", type=int, default=8, help="Hidden layer size (default: 8)"
    )
    train_parser.add_argument(
        "--output", type=int, default=1, help="Output size (default: 1)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=500, help="Training epochs (default: 500)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    train_parser.add_argument("--test", help="Test data file (.csv or .json)")
    train_parser.add_argument(
        "--save", default="model.npz", help="Save model to file (default: model.npz)"
    )

    # predict
    predict_parser = subparsers.add_parser(
        "predict", help="Run predictions with a trained model"
    )
    predict_parser.add_argument(
        "--model", required=True, help="Path to saved model (.npz)"
    )
    predict_parser.add_argument(
        "--input", help="Comma-separated input values (e.g. '1,0,1')"
    )
    predict_parser.add_argument(
        "--data", help="Data file to predict on (.csv or .json)"
    )

    # info
    info_parser = subparsers.add_parser(
        "info", help="Show model architecture and parameters"
    )
    info_parser.add_argument(
        "--model", required=True, help="Path to saved model (.npz)"
    )

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "info":
        cmd_info(args)
