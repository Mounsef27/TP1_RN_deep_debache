# exo1.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ---------- Utils ----------
def softmax_stable(S):
    # S: (tb, K)
    S_shift = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S_shift)
    return E / np.sum(E, axis=1, keepdims=True)

def forward(Xb, W, b):
    # Xb: (tb, d), W: (d, K), b: (1, K)
    S = Xb @ W + b
    Yhat = softmax_stable(S)
    return Yhat

def cross_entropy(Yhat, Ytrue, eps=1e-12):
    # mean CE over batch
    Yhat = np.clip(Yhat, eps, 1.0)
    return -np.mean(np.sum(Ytrue * np.log(Yhat), axis=1))

def accuracy(W, b, images, labels_onehot):
    pred = forward(images, W, b)
    return (pred.argmax(axis=1) == labels_onehot.argmax(axis=1)).mean() * 100.0

def plot_W_templates(W, filename="fig_exo1_W.png"):
    # W: (784,10). Show each column as 28x28 image.
    K = W.shape[1]
    plt.figure(figsize=(10, 3))
    for c in range(K):
        plt.subplot(2, 5, c + 1)
        plt.imshow(W[:, c].reshape(28, 28), cmap="seismic")
        plt.title(f"Classe {c}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# ---------- Training ----------
def main():
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784).astype("float32") / 255.0
    X_test  = X_test.reshape(10000, 784).astype("float32") / 255.0

    K = 10
    Y_train = to_categorical(y_train, K)
    Y_test  = to_categorical(y_test, K)

    N, d = X_train.shape

    # Params
    W = np.zeros((d, K), dtype=np.float32)
    b = np.zeros((1, K), dtype=np.float32)

    numEp = 20
    eta = 1e-1
    batch_size = 100
    nb_batches = N // batch_size

    # Logs
    loss_hist = []
    acc_train_hist = []
    acc_test_hist = []

    # Initial diagnostics
    init_loss = cross_entropy(forward(X_train[:1000], W, b), Y_train[:1000])
    print(f"Initial approx loss (on 1000): {init_loss:.4f} (should be ~2.3026)")

    rng = np.random.default_rng(0)

    for epoch in range(numEp):
        # Shuffle each epoch
        perm = rng.permutation(N)
        Xs = X_train[perm]
        Ys = Y_train[perm]

        epoch_loss = 0.0

        for bi in range(nb_batches):
            start = bi * batch_size
            end = start + batch_size
            Xb = Xs[start:end]
            Yb = Ys[start:end]

            # Forward
            Yhat = forward(Xb, W, b)

            # Loss
            batch_loss = cross_entropy(Yhat, Yb)
            epoch_loss += batch_loss

            # Backward (Eq. 4 & 5): dL/dS = (Yhat - Ytrue)/tb
            tb = Xb.shape[0]
            Delta = (Yhat - Yb)  # (tb,K)

            gradW = (Xb.T @ Delta) / tb          # (d,K)
            gradb = np.sum(Delta, axis=0, keepdims=True) / tb  # (1,K)

            # Update (Eq. 6 & 7)
            W -= eta * gradW
            b -= eta * gradb

        epoch_loss /= nb_batches

        # Metrics (compute on subsets for speed or full if you want)
        train_acc = accuracy(W, b, X_train, Y_train)
        test_acc  = accuracy(W, b, X_test, Y_test)

        loss_hist.append(epoch_loss)
        acc_train_hist.append(train_acc)
        acc_test_hist.append(test_acc)

        print(f"Epoch {epoch+1:02d}/{numEp} | loss={epoch_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%")

    # Figures: loss + accuracy
    plt.figure()
    plt.plot(np.arange(1, numEp+1), loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (cross-entropy)")
    plt.title("Exo1 - Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_exo1_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, numEp+1), acc_train_hist, label="Train")
    plt.plot(np.arange(1, numEp+1), acc_test_hist, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Exo1 - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_exo1_acc.png", dpi=200)
    plt.close()

    # Weight templates
    plot_W_templates(W, "fig_exo1_W.png")

    print("\nSaved figures:")
    print(" - fig_exo1_loss.png")
    print(" - fig_exo1_acc.png")
    print(" - fig_exo1_W.png")

if __name__ == "__main__":
    main()
