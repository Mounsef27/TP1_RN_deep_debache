# mlp.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# ---------- Activations ----------
def sigmoid(U):
    return 1.0 / (1.0 + np.exp(-U))

def softmax_stable(V):
    V = V - np.max(V, axis=1, keepdims=True)
    E = np.exp(V)
    return E / np.sum(E, axis=1, keepdims=True)

# ---------- Forward ----------
def forward(Xb, Wh, bh, Wy, by):
    # Xb: (tb,d)
    U = Xb @ Wh + bh          # (tb,L)
    H = sigmoid(U)            # (tb,L)
    V = H @ Wy + by           # (tb,K)
    Yhat = softmax_stable(V)  # (tb,K)
    return Yhat, H, U

def cross_entropy(Yhat, Ytrue, eps=1e-12):
    Yhat = np.clip(Yhat, eps, 1.0)
    return -np.mean(np.sum(Ytrue * np.log(Yhat), axis=1))

def accuracy_mlp(Wh, bh, Wy, by, X, Y):
    Yhat, _, _ = forward(X, Wh, bh, Wy, by)
    return (Yhat.argmax(axis=1) == Y.argmax(axis=1)).mean() * 100.0

# ---------- Initialization ----------
def init_params(d, L, K, mode="zero", sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    if mode == "zero":
        Wh = np.zeros((d, L), dtype=np.float32)
        bh = np.zeros((1, L), dtype=np.float32)
        Wy = np.zeros((L, K), dtype=np.float32)
        by = np.zeros((1, K), dtype=np.float32)
    elif mode == "normal":
        Wh = (sigma * rng.standard_normal((d, L))).astype(np.float32)
        bh = np.zeros((1, L), dtype=np.float32)
        Wy = (sigma * rng.standard_normal((L, K))).astype(np.float32)
        by = np.zeros((1, K), dtype=np.float32)
    elif mode == "xavier":
        # N(0,1)/sqrt(n_in) for each layer
        Wh = (rng.standard_normal((d, L)) / np.sqrt(d)).astype(np.float32)
        bh = np.zeros((1, L), dtype=np.float32)
        Wy = (rng.standard_normal((L, K)) / np.sqrt(L)).astype(np.float32)
        by = np.zeros((1, K), dtype=np.float32)
    else:
        raise ValueError("Unknown init mode.")
    return Wh, bh, Wy, by

# ---------- Training ----------
def train_mlp(X_train, Y_train, X_test, Y_test,
              L=100, eta=1.0, numEp=100, batch_size=100,
              init_mode="xavier", sigma=0.1, seed=0):

    N, d = X_train.shape
    K = Y_train.shape[1]
    Wh, bh, Wy, by = init_params(d, L, K, mode=init_mode, sigma=sigma, seed=seed)

    nb_batches = N // batch_size
    rng = np.random.default_rng(seed)

    loss_hist = []
    acc_train_hist = []
    acc_test_hist = []

    for epoch in range(numEp):
        perm = rng.permutation(N)
        Xs = X_train[perm]
        Ys = Y_train[perm]

        epoch_loss = 0.0

        for bi in range(nb_batches):
            Xb = Xs[bi*batch_size:(bi+1)*batch_size]
            Yb = Ys[bi*batch_size:(bi+1)*batch_size]
            tb = Xb.shape[0]

            # Forward
            Yhat, H, U = forward(Xb, Wh, bh, Wy, by)

            # Loss
            epoch_loss += cross_entropy(Yhat, Yb)

            # Backward
            # Delta_y = Yhat - Y*
            Delta_y = (Yhat - Yb)                          # (tb,K)

            # Grad Wy, by (Eq 8-9)
            gradWy = (H.T @ Delta_y) / tb                  # (L,K)
            gradby = np.sum(Delta_y, axis=0, keepdims=True) / tb  # (1,K)

            # Delta_h = (Delta_y Wy^T) ⊙ sigmoid'(U)
            # sigmoid'(U) = H ⊙ (1-H)
            Delta_h = (Delta_y @ Wy.T) * (H * (1.0 - H))    # (tb,L)

            # Grad Wh, bh
            gradWh = (Xb.T @ Delta_h) / tb                  # (d,L)
            gradbh = np.sum(Delta_h, axis=0, keepdims=True) / tb  # (1,L)

            # Update
            Wy -= eta * gradWy
            by -= eta * gradby
            Wh -= eta * gradWh
            bh -= eta * gradbh

        epoch_loss /= nb_batches

        train_acc = accuracy_mlp(Wh, bh, Wy, by, X_train, Y_train)
        test_acc  = accuracy_mlp(Wh, bh, Wy, by, X_test, Y_test)

        loss_hist.append(epoch_loss)
        acc_train_hist.append(train_acc)
        acc_test_hist.append(test_acc)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[{init_mode}] Epoch {epoch+1:03d}/{numEp} | loss={epoch_loss:.4f} | train={train_acc:.2f}% | test={test_acc:.2f}%")

    return (Wh, bh, Wy, by), (loss_hist, acc_train_hist, acc_test_hist)

def save_curves(loss_hist, acc_train_hist, acc_test_hist, prefix):
    E = len(loss_hist)

    plt.figure()
    plt.plot(np.arange(1, E+1), loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (cross-entropy)")
    plt.title(f"MLP - Loss ({prefix})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fig_mlp_loss_{prefix}.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, E+1), acc_train_hist, label="Train")
    plt.plot(np.arange(1, E+1), acc_test_hist, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"MLP - Accuracy ({prefix})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fig_mlp_acc_{prefix}.png", dpi=200)
    plt.close()

def main():
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784).astype("float32") / 255.0
    X_test  = X_test.reshape(10000, 784).astype("float32") / 255.0

    K = 10
    Y_train = to_categorical(y_train, K)
    Y_test  = to_categorical(y_test, K)

    # Hyperparams TP
    L = 100
    eta = 1.0
    numEp = 100
    batch_size = 100

    results = []

    # 1) Zero init
    params0, hist0 = train_mlp(X_train, Y_train, X_test, Y_test,
                               L=L, eta=eta, numEp=numEp, batch_size=batch_size,
                               init_mode="zero", seed=0)
    save_curves(*hist0, prefix="zero")
    acc0 = hist0[2][-1]
    results.append(("Initialisation à zéro", acc0))

    # 2) Normal init
    sigma = 0.1
    paramsN, histN = train_mlp(X_train, Y_train, X_test, Y_test,
                               L=L, eta=eta, numEp=numEp, batch_size=batch_size,
                               init_mode="normal", sigma=sigma, seed=0)
    save_curves(*histN, prefix=f"normal_sigma{sigma}")
    accN = histN[2][-1]
    results.append((f"Loi normale (σ={sigma})", accN))

    # 3) Xavier init
    paramsX, histX = train_mlp(X_train, Y_train, X_test, Y_test,
                               L=L, eta=eta, numEp=numEp, batch_size=batch_size,
                               init_mode="xavier", seed=0)
    save_curves(*histX, prefix="xavier")
    accX = histX[2][-1]
    results.append(("Xavier", accX))

    print("\n=== Résumé accuracies test finales ===")
    for name, acc in results:
        print(f"{name:25s} : {acc:.2f}%")

    print("\nSaved figures:")
    print(" - fig_mlp_loss_zero.png, fig_mlp_acc_zero.png")
    print(" - fig_mlp_loss_normal_sigma0.1.png, fig_mlp_acc_normal_sigma0.1.png")
    print(" - fig_mlp_loss_xavier.png, fig_mlp_acc_xavier.png")

if __name__ == "__main__":
    main()
