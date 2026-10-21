"""
train_model.py
---------------------------------------
Capstone showdown: Train multiple model types (OvR Logistic, Softmax, RBF, MLP)
on Fashion-MNIST using the my_ml_lib package.
"""

import os
import pickle
import argparse
import numpy as np

from my_ml_lib.datasets._loaders import load_fashion_mnist
from my_ml_lib.model_selection._split import train_test_val_split
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.preprocessing._gaussian import GaussianBasisFeatures
from my_ml_lib.linear_models.classification._logistic import LogisticRegression

from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.losses import CrossEntropyLoss
from my_ml_lib.nn.optim import SGD
from my_ml_lib.nn.autograd import Value


# ==========================================================
# Helper: One-vs-Rest Logistic Regression training
# ==========================================================
def ovr_fit_predict(LogRegCls, X_tr, y_tr, X_val, alpha, max_iter):
    """Trains OvR logistic classifiers, returns predictions + models."""
    classes = np.unique(y_tr)
    models = []

    for c in classes:
        y_bin = (y_tr == c).astype(int)
        clf = LogRegCls(alpha=alpha, max_iter=max_iter)  # removed lr argument
        clf.fit(X_tr, y_bin)
        models.append(clf)

    # Validation predictions
    probs = [clf.predict_proba(X_val)[:, 1].reshape(-1, 1) for clf in models]
    probs = np.hstack(probs)
    y_val_pred = np.argmax(probs, axis=1)

    return y_val_pred, models



# ==========================================================
# Training loops for autograd models
# ==========================================================
def train_autograd_model(model, X_tr, y_tr, X_val, y_val, lr, epochs):
    """Generic trainer for autograd-based models."""
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        logits = model(Value(X_tr))
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Validation
        val_logits = model(Value(X_val))
        y_val_pred = np.argmax(val_logits.data, axis=1)
        val_acc = np.mean(y_val_pred == y_val)

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Val acc={val_acc:.4f} | Loss={loss.data:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.copy() for k, v in model.state_dict().items()}

    print(f"✅ Best Validation Accuracy: {best_val_acc:.4f}")
    return best_state, best_val_acc


# ==========================================================
# Main entry
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Capstone showdown trainer")
    parser.add_argument("--model", type=str, required=True,
                        choices=["logistic", "softmax_raw", "softmax_rbf", "mlp"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-centers", type=int, default=60)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--mark-best", action="store_true",
                        help="Also copy best model to saved_models/best_model.npz or .pkl")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ======================================================
    # Load and Split Data
    # ======================================================
    X, y = load_fashion_mnist(data_folder="data", normalize=True)
    X_tr, X_val, X_test, y_tr, y_val, y_test = train_test_val_split(
        X, y, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True, random_state=42
    )
    print(f"✅ Data split: train={X_tr.shape}, val={X_val.shape}, test={X_test.shape}")

    # ======================================================
    # Model Branches
    # ======================================================

    # 1️⃣ OvR Logistic Regression
    if args.model == "logistic":
        print("\n=== OvR Logistic Regression ===")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        y_val_pred, models = ovr_fit_predict(LogisticRegression, X_tr, y_tr, X_val,
                                             alpha=args.alpha, max_iter=args.epochs)
        val_acc = np.mean(y_val_pred == y_val)
        print(f"✅ Validation Accuracy: {val_acc:.4f}")

        # Save model
        path = os.path.join(args.save_dir, "best_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"models": models, "scaler": scaler, "alpha": args.alpha}, f)
        print(f"💾 Saved OvR Logistic model(s) to {path}")

        return

    # 2️⃣ Softmax Regression (Raw Pixels)
    elif args.model == "softmax_raw":
        print("\n=== Softmax Regression (Raw) ===")
        n_features = X_tr.shape[1]
        n_classes = len(np.unique(y_tr))

        model = Sequential(Linear(n_features, n_classes))
        best_state, _ = train_autograd_model(model, X_tr, y_tr, X_val, y_val, args.lr, args.epochs)

        # Save model
        model.load_state_dict(os.path.join(args.save_dir, "softmax_raw.npz"))
        print(f"💾 Saved to {os.path.join(args.save_dir, 'softmax_raw.npz')}")
        if args.mark_best:
            model.save_state_dict(os.path.join(args.save_dir, "best_model.npz"))

    elif args.model == "softmax_poly":
        from my_ml_lib.preprocessing._polynomial import PolynomialFeatures
        print("\n=== Softmax + Polynomial Features ===")
        poly = PolynomialFeatures(degree=2)
        X_tr_p = poly.fit_transform(X_tr)
        X_val_p = poly.transform(X_val)

        n_features = X_tr_p.shape[1]
        n_classes = len(np.unique(y_tr))

        model = Sequential(Linear(n_features, n_classes))
        best_state, _ = train_autograd_model(model, X_tr_p, y_tr, X_val_p, y_val, args.lr, args.epochs)

        path_specific = os.path.join(args.save_dir, "softmax_poly.npz")
        model.save_state_dict(path_specific)
        print(f"💾 Saved to {path_specific}")

    # 3️⃣ Softmax + Gaussian Basis Features
    elif args.model == "softmax_rbf":
        print("\n=== Softmax + Gaussian Basis Features ===")
        rbf = GaussianBasisFeatures(n_centers=args.n_centers, sigma=args.sigma, random_state=42)
        X_tr_r = rbf.fit_transform(X_tr)
        X_val_r = rbf.transform(X_val)

        n_features = X_tr_r.shape[1]
        n_classes = len(np.unique(y_tr))

        model = Sequential(Linear(n_features, n_classes))
        best_state, _ = train_autograd_model(model, X_tr_r, y_tr, X_val_r, y_val, args.lr, args.epochs)

        path_specific = os.path.join(args.save_dir, "softmax_rbf.npz")
        model.save_state_dict(path_specific)
        print(f"💾 Saved to {path_specific}")

        if args.mark_best:
            model.save_state_dict(os.path.join(args.save_dir, "best_model.npz"))

    # 4️⃣ MLP (Multi-Layer Perceptron)
    elif args.model == "mlp":
        print("\n=== MLP (Autograd) ===")
        n_features = X_tr.shape[1]
        n_classes = len(np.unique(y_tr))

        model = Sequential(
            Linear(n_features, args.hidden),
            ReLU(),
            Linear(args.hidden, 128),
            ReLU(),
            Linear(128, n_classes)
        )

        best_state, _ = train_autograd_model(model, X_tr, y_tr, X_val, y_val, args.lr, args.epochs)

        path_specific = os.path.join(args.save_dir, "mlp_best.npz")
        model.save_state_dict(path_specific)
        print(f"💾 Saved to {path_specific}")

        if args.mark_best:
            best_path = os.path.join(args.save_dir, "best_model.npz")
            model.save_state_dict(best_path)
            print(f"🏷️  Also saved for submission: {best_path}")

    else:
        raise ValueError(f"Unknown model type: {args.model}")


# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    main()
