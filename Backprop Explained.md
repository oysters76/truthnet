# How Your Neural Network Learns

A visual walkthrough of feedforward, backprop, and why transpose makes gradient descent work.

## 1. Your Network's Shape

Your network has three layers of weights. Data flows left to right: 3 inputs, through two hidden layers of 4 neurons each, down to 1 output.

Your network: 3 inputs → 4 hidden → 4 hidden → 1 output. Each weight matrix shape is (inputs to that layer, outputs from that layer).

---

## 2. The Forward Pass

At each layer, two things happen in sequence: a **linear transformation** (matmul + bias) and then an **activation** (ReLU). The output layer skips ReLU.

Each layer: multiply input by weights, add bias, then activate with ReLU. The output layer has no ReLU.

The key operation at every layer is the same **linear transformation**:

```
z = input @ W + b
```

This is a matrix multiplication. The input's columns must match the weight matrix's rows — that's what makes the shapes compatible. Then ReLU zeroes out any negatives, giving the network its ability to learn non-linear patterns.

---

## 3. Why We Need Transpose

This is the core insight of backprop. In the forward pass, you computed `z = input @ W`. Now you need to go *backwards* — given how wrong the output was (the gradient), figure out how to adjust each weight.

There are two questions at each layer:

1. **How should I change W?** → gives us `dW`
2. **How much did the input contribute?** → gives us `d(input)`, to pass blame further back

The answer to both involves a transpose. Here's why.

### The shape constraint

The gradient for a matrix **must be the same shape as the matrix itself**. If `W` is `(3,4)`, then `dW` must also be `(3,4)` — each gradient value tells one specific weight how much to change.

**The rule:** When you need to multiply two things and get a specific output shape, the transpose is what makes the inner dimensions line up.

### Computing dW — "how to update the weights"

Forward was: `input @ W → output`, with shapes `(1,3) @ (3,4) → (1,4)`.

We have the gradient at the output: `dz` with shape `(1,4)`. We need `dW` with shape `(3,4)`.

The transpose flips input from (1,3) to (3,1), making the inner dimensions (1 and 1) match, and the outer dimensions give us (3,4) — exactly W's shape.

Without the transpose, you'd have `(1,3) @ (1,4)` — the inner dimensions are 3 and 1, which don't match. The matmul can't even run. The transpose is the only way to make the shapes produce a result that matches W.

### Computing d(input) — "passing the blame backwards"

Now the other direction: how much did each input neuron contribute to the error? We need `d(input)` with shape `(1,3)` — same shape as the input.

Transposing W from (3,4) to (4,3) makes the inner dimensions (4 and 4) match, and the result is (1,3) — same shape as the input.

**Think of it this way:** Transpose is not some special calculus trick. It's the only arrangement that makes the matrix multiplication produce a result with the correct shape. The math of the chain rule *requires* the result to match the original matrix's shape, and transpose is what gets you there.

### The complete rule

For every layer where the forward pass was `z = input @ W + b`:

| What you want | Formula | Operation | Why this shape |
| --- | --- | --- | --- |
| `dW` | `input.T @ dz` | matmul | Result must match W's shape |
| `db` | `dz` | copy | Bias shifts output directly |
| `d(input)` | `dz @ W.T` | matmul | Result must match input's shape |
| `dz` through ReLU | `d(activation) * relu'(z)` | element-wise `*` | ReLU acts per-element |

---

## 4. Forward and Backward Are Mirrors

This is the beautiful connection. Every linear transformation in the forward pass has a *mirror image* in the backward pass. The same weight matrix appears in both — just transposed.

The backward pass reverses every step. The same W appears transposed, and ReLU becomes a gate that either passes or blocks the gradient.

Notice the symmetry:

- **Forward:** data flows through `W` → the weights *transform* the input
- **Backward:** gradient flows through `W.T` → the same weights *distribute the blame* back to the input
- **ReLU forward:** zeros out negatives → **ReLU backward:** blocks gradient at the same positions
- **Bias forward:** adds a constant → **Bias backward:** gradient passes straight through (adding a constant doesn't change the slope)

The linear transformation and its backprop are the *same operation* — just with the matrix flipped. That's not a coincidence. It's because the derivative of `a @ B` with respect to `a` is `B.T`, by the rules of matrix calculus. The transpose *is* the derivative.

---

## 5. The Full Backprop Chain

Here's your entire `backprop` method, with each line mapped to what it does. The gradient starts at the loss and flows backwards through every layer.

Gradient flows top to bottom. At each layer: gate through ReLU (element-wise), compute weight gradient (matmul with transpose), pass blame to previous layer (matmul with transpose).

Every layer does the same three things:

1. **Gate through ReLU** (`*` element-wise) — dead neurons block the gradient
2. **Compute dW** (`input.T @ dz`) — how to update this layer's weights
3. **Pass blame back** (`dz @ W.T`) — send gradient to the previous layer

The output layer skips step 1 because there's no ReLU on the output.

---

## 6. The Training Loop

All of the above happens inside a single training step. Your `optimize` method repeats this loop for every example in the training set, over many epochs:

The training loop: predict, measure error, compute gradients, nudge weights. Repeat hundreds of times until the loss is small.

Each time around the loop, the weights get a tiny nudge in the direction that reduces the error. After enough loops, the network has learned the pattern.

```
self.W1 -= learning_rate * dW1   # nudge, don't replace
self.W2 -= learning_rate * dW2
self.W3 -= learning_rate * dW3
```

The `learning_rate` (0.01) controls how big each nudge is. Too big and the network overshoots; too small and it learns slowly.

---

## Quick Reference

Your complete `backprop` method, annotated:

```
# Start: derivative of loss (y - ŷ)²
dL = -2 * (y - y_hat)

# Layer 3 (output) — no ReLU
dW3 = a2.T @ dL          # matmul: weight gradient
db3 = dL                  # copy: bias gradient

# Layer 2 — pass through W3, gate through ReLU
da2 = dL @ self.W3.T      # matmul: blame to layer 2
dz2 = da2 * relu'(z2)     # elem: gate dead neurons
dW2 = a1.T @ dz2          # matmul: weight gradient
db2 = dz2                 # copy: bias gradient

# Layer 1 — pass through W2, gate through ReLU
da1 = dz2 @ self.W2.T     # matmul: blame to layer 1
dz1 = da1 * relu'(z1)     # elem: gate dead neurons
dW1 = x.T @ dz1           # matmul: weight gradient
db1 = dz1                 # copy: bias gradient
```

**The two operations, one last time:**

`@` **matmul** — used when data passed through a weight matrix. Transpose makes the shapes fit.

`*` **element-wise** — used only for ReLU, because it acts on each number independently.