import numpy as np

from truthnet.network import Network


def test_init_shapes():
    net = Network(3, 4, 1)
    assert net.W1.shape == (3, 4)
    assert net.b1.shape == (1, 4)
    assert net.W2.shape == (4, 4)
    assert net.b2.shape == (1, 4)
    assert net.W3.shape == (4, 1)
    assert net.b3.shape == (1, 1)


def test_feedforward_output_shape():
    net = Network(3, 4, 1)
    z1, a1, z2, a2, y_hat = net.feedforward(np.array([1, 0, 1]))
    assert z1.shape == (1, 4)
    assert a1.shape == (1, 4)
    assert z2.shape == (1, 4)
    assert a2.shape == (1, 4)
    assert y_hat.shape == (1, 1)


def test_feedforward_multi_output():
    net = Network(2, 8, 3)
    _, _, _, _, y_hat = net.feedforward(np.array([1, 0]))
    assert y_hat.shape == (1, 3)


def test_predict():
    net = Network(3, 4, 1)
    y_hat = net.predict(np.array([1, 0, 1]))
    assert y_hat.shape == (1, 1)


def test_relu_derivative():
    net = Network(2, 2, 1)
    result = net.relu_derivative(np.array([-1, 0, 0.5, 3]))
    np.testing.assert_array_equal(result, [0, 0, 1, 1])


def test_backprop_gradient_shapes():
    net = Network(3, 4, 1)
    x = np.array([1, 0, 1])
    z1, a1, z2, a2, y_hat = net.feedforward(x)
    dW3, db3, dW2, db2, dW1, db1 = net.backprop(x, 1.0, y_hat, z1, a1, z2, a2)
    assert dW3.shape == net.W3.shape
    assert db3.shape == net.b3.shape
    assert dW2.shape == net.W2.shape
    assert db2.shape == net.b2.shape
    assert dW1.shape == net.W1.shape
    assert db1.shape == net.b1.shape


def test_training_reduces_loss():
    np.random.seed(42)
    net = Network(3, 8, 1)
    train_set = [
        ([1, 0, 0], 1),
        ([0, 1, 0], 0),
        ([1, 0, 1], 1),
        ([0, 1, 1], 0),
    ]

    initial_loss = sum(
        net.L(y, net.predict(np.array(ix))).item() for ix, y in train_set
    )

    net.optimize(train_set, learning_rate=0.01, epochs=200, log_every=9999)

    final_loss = sum(net.L(y, net.predict(np.array(ix))).item() for ix, y in train_set)

    assert final_loss < initial_loss
    assert final_loss < 0.1


def test_save_and_load(tmp_path):
    net = Network(3, 4, 1)
    x = np.array([1, 0, 1])
    original_pred = net.predict(x)

    path = tmp_path / "model.npz"
    net.save(str(path))

    loaded = Network.load(str(path))
    assert loaded.input_size == 3
    assert loaded.hidden_size == 4
    assert loaded.output_size == 1
    np.testing.assert_array_equal(loaded.predict(x), original_pred)
