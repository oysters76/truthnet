import numpy as np


class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.random.rand(1, hidden_size)

        self.W2 = np.random.rand(hidden_size, hidden_size)
        self.b2 = np.random.rand(1, hidden_size)

        self.W3 = np.random.rand(hidden_size, output_size)
        self.b3 = np.random.rand(1, output_size)

    def feedforward(self, x):
        z1 = np.matmul(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)

        z2 = np.matmul(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)

        y_hat = np.matmul(a2, self.W3) + self.b3

        return z1, a1, z2, a2, y_hat

    def predict(self, x):
        _, _, _, _, y_hat = self.feedforward(x)
        return y_hat

    def L(self, y, y_hat):
        return (y - y_hat) ** 2

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def backprop(self, x, y, y_hat, z1, a1, z2, a2):
        x = x.reshape(1, -1)

        dL = -2 * (y - y_hat)
        dW3 = a2.T @ dL
        db3 = dL

        da2 = dL @ self.W3.T
        dz2 = da2 * self.relu_derivative(z2)
        dW2 = a1.T @ dz2
        db2 = dz2

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = x.T @ dz1
        db1 = dz1

        return dW3, db3, dW2, db2, dW1, db1

    def optimize(self, train_set, learning_rate=0.01, epochs=500, log_every=100):
        for epoch in range(epochs):
            for ix, y in train_set:
                x = np.array(ix)
                z1, a1, z2, a2, y_hat = self.feedforward(x)
                dW3, db3, dW2, db2, dW1, db1 = self.backprop(
                    x, y, y_hat, z1, a1, z2, a2
                )

                self.W1 -= learning_rate * dW1
                self.W2 -= learning_rate * dW2
                self.W3 -= learning_rate * dW3

                self.b1 -= learning_rate * db1
                self.b2 -= learning_rate * db2
                self.b3 -= learning_rate * db3

            if (epoch + 1) % log_every == 0:
                total_loss = sum(
                    self.L(y, self.predict(np.array(ix))).item() for ix, y in train_set
                )
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.6f}")

    def save(self, path):
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        net = cls(
            int(data["input_size"]),
            int(data["hidden_size"]),
            int(data["output_size"]),
        )
        net.W1 = data["W1"]
        net.b1 = data["b1"]
        net.W2 = data["W2"]
        net.b2 = data["b2"]
        net.W3 = data["W3"]
        net.b3 = data["b3"]
        return net
