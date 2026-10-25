import json
import subprocess
import sys


def run_cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "truthnet", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def test_train_csv(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("x1,x2,x3,y\n1,0,0,1\n0,1,0,0\n1,0,1,1\n")
    model_path = tmp_path / "model.npz"

    result = run_cli(
        "train", "--data", str(csv_file), "--epochs", "50", "--save", str(model_path)
    )
    assert result.returncode == 0
    assert "Model saved" in result.stdout
    assert model_path.exists()


def test_train_json(tmp_path):
    json_file = tmp_path / "train.json"
    json_file.write_text(json.dumps([[[1, 0], 1], [[0, 1], 0]]))
    model_path = tmp_path / "model.npz"

    result = run_cli(
        "train", "--data", str(json_file), "--epochs", "50", "--save", str(model_path)
    )
    assert result.returncode == 0
    assert model_path.exists()


def test_train_with_test_set(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("1,0,1\n0,1,0\n")
    test.write_text("1,1,1\n0,0,0\n")
    model_path = tmp_path / "model.npz"

    result = run_cli(
        "train",
        "--data",
        str(train),
        "--epochs",
        "50",
        "--test",
        str(test),
        "--save",
        str(model_path),
    )
    assert result.returncode == 0
    assert "Test Results" in result.stdout


def test_predict_single_input(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("1,0,1\n0,1,0\n")
    model_path = tmp_path / "model.npz"

    run_cli(
        "train", "--data", str(csv_file), "--epochs", "50", "--save", str(model_path)
    )

    result = run_cli("predict", "--model", str(model_path), "--input", "1,0")
    assert result.returncode == 0
    assert "Prediction:" in result.stdout


def test_predict_from_file(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("1,0,1\n0,1,0\n")
    model_path = tmp_path / "model.npz"

    run_cli(
        "train", "--data", str(csv_file), "--epochs", "50", "--save", str(model_path)
    )

    result = run_cli("predict", "--model", str(model_path), "--data", str(csv_file))
    assert result.returncode == 0
    assert "Predictions" in result.stdout


def test_info(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("1,0,1\n0,1,0\n")
    model_path = tmp_path / "model.npz"

    run_cli(
        "train", "--data", str(csv_file), "--epochs", "10", "--save", str(model_path)
    )

    result = run_cli("info", "--model", str(model_path))
    assert result.returncode == 0
    assert "Architecture:" in result.stdout
    assert "Total parameters:" in result.stdout


def test_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "info" in result.stdout
