import json

import pytest

from truthnet.data import infer_sizes, load_data


def test_load_csv_with_header(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x1,x2,y\n1,0,1\n0,1,0\n")

    data = load_data(str(csv_file))
    assert len(data) == 2
    assert data[0] == ([1.0, 0.0], 1.0)
    assert data[1] == ([0.0, 1.0], 0.0)


def test_load_csv_without_header(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("1,0,1\n0,1,0\n")

    data = load_data(str(csv_file))
    assert len(data) == 2
    assert data[0] == ([1.0, 0.0], 1.0)


def test_load_json(tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text(
        json.dumps(
            [
                [[1, 0], 1],
                [[0, 1], 0],
            ]
        )
    )

    data = load_data(str(json_file))
    assert len(data) == 2
    assert data[0] == ([1, 0], 1)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.csv")


def test_load_unsupported_format(tmp_path):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("1,0,1\n")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_data(str(txt_file))


def test_infer_sizes():
    data = [([1, 0, 0], 1), ([0, 1, 0], 0)]
    input_size, output_size = infer_sizes(data)
    assert input_size == 3
    assert output_size == 1


def test_infer_sizes_multi_output():
    data = [([1, 0], [1, 0, 0]), ([0, 1], [0, 1, 0])]
    input_size, output_size = infer_sizes(data)
    assert input_size == 2
    assert output_size == 3
