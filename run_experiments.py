# run_experiments.py
import os
import pickle
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy

# import helpers + modules from your package
from my_ml_lib.datasets._loaders import load_fashion_mnist
from my_ml_lib.model_selection._split import train_test_val_split
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.preprocessing._polynomial import PolynomialFeatures
from my_ml_lib.preprocessing._gaussian import GaussianBasisFeatures
from my_ml_lib.linear_models.classification._logistic import LogisticRegression

from my_ml_lib.nn.modules.linear import Linear
from my_ml_lib.nn.modules.activations import ReLU, Sigmoid
from my_ml_lib.nn.modules.containers import Sequential
from my_ml_lib.nn.losses import CrossEntropyLoss
from my_ml_lib.nn.optim import SGD
from my_ml_lib.nn.autograd import Value

# reuse your train loop
def train_autograd_model(model, X_tr, y_tr, X_val, y_val, lr, epochs):
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

        if epoch % 5 == 0 or epoch == 1:
            print(f" Epoch {epoch:3d} | Val acc={val_acc:.4f} | Loss={loss.data:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # capture state dict
            best_state = {k: v.copy() for k, v in model.state_dict().items()}

    return best_state, best_val_acc

def ovr_fit_predict(LogRegCls, X_tr, y_tr, X_val, alpha, max_iter):
    classes = np.unique(y_tr)
    models = []
    for c in classes:
        y_bin = (y_tr == c).astype(int)
        clf = LogRegCls(alpha=alpha, max_iter=max_iter)
        clf.fit(X_tr, y_bin)
        models.append(clf)
    # assemble probs: each clf.predict_proba returns 2-column [p0, p1]
    probs = [m.predict_proba(X_val)[:,1].reshape(-1,1) for m in models]
    probs = np.hstack(probs)
    y_val_pred = np.argmax(probs, axis=1)
    return y_val_pred, models

# Utility to evaluate autograd model on data using a saved state
def eval_autograd_model_with_state(model, state, X, y):
    # try load_state_dict accepting dict; else set attributes manually
    try:
        model.load_state_dict(state)
    except Exception:
        # fallback: assign weights if state keys follow 'layer.weight' style
        sd = state
        for k, v in sd.items():
            setattr(model, k, v)
    logits = model(Value(X))
    y_pred = np.argmax(logits.data, axis=1)
    return np.mean(y_pred == y)

def save_autograd_state(state, path):
    # state is dict of numpy arrays
    np.savez(path, **state)

def main():
    out_dir = "experiment_results"
    os.makedirs(out_dir, exist_ok=True)
    saved_models_dir = os.path.join(out_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    # 1) Load and split
    X, y = load_fashion_mnist(data_folder="data", normalize=True)
    X_tr, X_val, X_test, y_tr, y_val, y_test = train_test_val_split(
        X, y, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True, random_state=42
    )
    print(f"Data shapes: train={X_tr.shape}, val={X_val.shape}, test={X_test.shape}")

    results = []

    # ---------------------------
    # 1) OvR Logistic (IRLS)
    # ---------------------------
    print("\n=== Grid search: OvR Logistic ===")
    alphas = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
    max_iters = [10, 50, 100]
    best = {"val_acc": -1}
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    for alpha, mx in itertools.product(alphas, max_iters):
        y_val_pred, models = ovr_fit_predict(LogisticRegression, X_tr_s, y_tr, X_val_s,
                                             alpha=alpha, max_iter=mx)
        val_acc = np.mean(y_val_pred == y_val)
        print(f" alpha={alpha:.4g} max_iter={mx} -> val_acc={val_acc:.4f}")
        if val_acc > best["val_acc"]:
            best.update({"val_acc": val_acc, "alpha": alpha, "max_iter": mx, "models": models, "scaler": scaler})

    # Evaluate on test using chosen best
    models = best["models"]
    probs = [m.predict_proba(X_test)[:,1].reshape(-1,1) for m in models]
    probs = np.hstack(probs)
    y_test_pred = np.argmax(probs, axis=1)
    test_acc = np.mean(y_test_pred == y_test)
    print(f"OvR Logistic best val={best['val_acc']:.4f} alpha={best['alpha']} -> test_acc={test_acc:.4f}")
    # save pickle
    with open(os.path.join(saved_models_dir, "ovr_logistic_best.pkl"), "wb") as f:
        pickle.dump({"models": best["models"], "scaler": best["scaler"], "alpha": best["alpha"], "max_iter": best["max_iter"]}, f)
    results.append(("OvR_Logistic", best["val_acc"], test_acc, f"alpha={best['alpha']}, max_iter={best['max_iter']}"))

    # ---------------------------
    # 2) Softmax (Raw)
    # ---------------------------
    print("\n=== Grid search: Softmax Raw ===")
    lrs = [1e-2, 5e-3, 1e-3]
    epochs_list = [20, 50]
    best = {"val_acc": -1}
    for lr, epochs in itertools.product(lrs, epochs_list):
        n_features = X_tr.shape[1]
        n_classes = len(np.unique(y_tr))
        model = Sequential(Linear(n_features, n_classes))
        state, val_acc = train_autograd_model(model, X_tr, y_tr, X_val, y_val, lr, epochs)
        print(f" lr={lr} epochs={epochs} -> val_acc={val_acc:.4f}")
        if val_acc > best["val_acc"]:
            best.update({"val_acc": val_acc, "lr": lr, "epochs": epochs, "state": state})

    # evaluate best on test
    model_best = Sequential(Linear(X_tr.shape[1], len(np.unique(y_tr))))
    if best["state"] is not None:
        try:
            model_best.load_state_dict(best["state"])
        except Exception:
            pass
    test_acc = eval_autograd_model_with_state(model_best, best["state"], X_test, y_test)
    print(f"Softmax Raw best val={best['val_acc']:.4f} -> test_acc={test_acc:.4f}")
    save_autograd_state(best["state"], os.path.join(saved_models_dir, "softmax_raw_best.npz"))
    results.append(("Softmax_Raw", best["val_acc"], test_acc, f"lr={best['lr']}, epochs={best['epochs']}"))

    # ---------------------------
    # 3) Softmax + Polynomial
    # ---------------------------
    print("\n=== Grid search: Softmax + Polynomial ===")
    degrees = [2, 3]                       # keep small to avoid explosion
    lrs = [1e-2, 5e-3]
    epochs_list = [20, 40]
    best = {"val_acc": -1}
    for deg, lr, epochs in itertools.product(degrees, lrs, epochs_list):
        poly = PolynomialFeatures(degree=deg)
        X_tr_p = poly.fit_transform(X_tr)
        X_val_p = poly.transform(X_val)

        n_features = X_tr_p.shape[1]
        model = Sequential(Linear(n_features, len(np.unique(y_tr))))
        state, val_acc = train_autograd_model(model, X_tr_p, y_tr, X_val_p, y_val, lr, epochs)
        print(f" deg={deg} lr={lr} epochs={epochs} -> val_acc={val_acc:.4f}")
        if val_acc > best["val_acc"]:
            best.update({"val_acc": val_acc, "deg": deg, "lr": lr, "epochs": epochs, "state": state, "poly": deepcopy(poly)})

    # evaluate best on test
    model_best = Sequential(Linear(best["poly"].transform(X_tr).shape[1], len(np.unique(y_tr))))
    try:
        model_best.load_state_dict(best["state"])
    except Exception:
        pass
    X_test_p = best["poly"].transform(X_test)
    test_acc = eval_autograd_model_with_state(model_best, best["state"], X_test_p, y_test)
    print(f"Softmax Poly best val={best['val_acc']:.4f} -> test_acc={test_acc:.4f}")
    save_autograd_state(best["state"], os.path.join(saved_models_dir, "softmax_poly_best.npz"))
    # save poly transformer via pickle
    with open(os.path.join(saved_models_dir, "softmax_poly_transformer.pkl"), "wb") as f:
        pickle.dump(best["poly"], f)
    results.append(("Softmax_Poly", best["val_acc"], test_acc, f"deg={best['deg']}, lr={best['lr']}, epochs={best['epochs']}"))

    # ---------------------------
    # 4) Softmax + Gaussian (RBF)
    # ---------------------------
    print("\n=== Grid search: Softmax + RBF ===")
    n_centers_list = [30, 60]   # try small sets (60 is reasonable)
    sigmas = [0.5, 1.0, 2.0]
    lrs = [1e-2, 5e-3]
    epochs_list = [20, 40]
    best = {"val_acc": -1}
    for n_centers, sigma, lr, epochs in itertools.product(n_centers_list, sigmas, lrs, epochs_list):
        rbf = GaussianBasisFeatures(n_centers=n_centers, sigma=sigma, random_state=42)
        X_tr_r = rbf.fit_transform(X_tr)
        X_val_r = rbf.transform(X_val)

        model = Sequential(Linear(X_tr_r.shape[1], len(np.unique(y_tr))))
        state, val_acc = train_autograd_model(model, X_tr_r, y_tr, X_val_r, y_val, lr, epochs)
        print(f" centers={n_centers} sigma={sigma} lr={lr} epochs={epochs} -> val_acc={val_acc:.4f}")
        if val_acc > best["val_acc"]:
            best.update({"val_acc": val_acc, "n_centers": n_centers, "sigma": sigma, "lr": lr, "epochs": epochs, "state": state, "rbf": deepcopy(rbf)})

    # evaluate best on test
    model_best = Sequential(Linear(best["rbf"].transform(X_tr).shape[1], len(np.unique(y_tr))))
    try:
        model_best.load_state_dict(best["state"])
    except Exception:
        pass
    X_test_r = best["rbf"].transform(X_test)
    test_acc = eval_autograd_model_with_state(model_best, best["state"], X_test_r, y_test)
    print(f"Softmax RBF best val={best['val_acc']:.4f} -> test_acc={test_acc:.4f}")
    save_autograd_state(best["state"], os.path.join(saved_models_dir, "softmax_rbf_best.npz"))
    with open(os.path.join(saved_models_dir, "softmax_rbf_transformer.pkl"), "wb") as f:
        pickle.dump(best["rbf"], f)
    results.append(("Softmax_RBF", best["val_acc"], test_acc, f"n_centers={best['n_centers']}, sigma={best['sigma']}, lr={best['lr']}, epochs={best['epochs']}"))

    # ---------------------------
    # 5) MLP
    # ---------------------------
    print("\n=== Grid search: MLP ===")
    mlp_archs = [
        [256],
        [512],
        [256, 128],
    ]
    lrs = [1e-3, 5e-4]
    epochs_list = [30, 50]
    best = {"val_acc": -1}
    for arch, lr, epochs in itertools.product(mlp_archs, lrs, epochs_list):
        # build MLP sequentially
        layers = []
        input_dim = X_tr.shape[1]
        for h in arch:
            layers.append(Linear(input_dim, h))
            layers.append(ReLU())
            input_dim = h
        layers.append(Linear(input_dim, len(np.unique(y_tr))))
        model = Sequential(*layers)

        state, val_acc = train_autograd_model(model, X_tr, y_tr, X_val, y_val, lr, epochs)
        print(f" arch={arch} lr={lr} epochs={epochs} -> val_acc={val_acc:.4f}")
        if val_acc > best["val_acc"]:
            best.update({"val_acc": val_acc, "arch": arch, "lr": lr, "epochs": epochs, "state": state})

    # evaluate best on test
    # rebuild best model
    layers = []
    input_dim = X_tr.shape[1]
    for h in best["arch"]:
        layers.append(Linear(input_dim, h))
        layers.append(ReLU())
        input_dim = h
    layers.append(Linear(input_dim, len(np.unique(y_tr))))
    model_best = Sequential(*layers)
    try:
        model_best.load_state_dict(best["state"])
    except Exception:
        pass
    test_acc = eval_autograd_model_with_state(model_best, best["state"], X_test, y_test)
    print(f"MLP best val={best['val_acc']:.4f} arch={best['arch']} -> test_acc={test_acc:.4f}")
    save_autograd_state(best["state"], os.path.join(saved_models_dir, "mlp_best.npz"))
    results.append(("MLP", best["val_acc"], test_acc, f"arch={best['arch']}, lr={best['lr']}, epochs={best['epochs']}"))

    # ---------------------------
    # Save summary
    # ---------------------------
    df = pd.DataFrame(results, columns=["model", "val_acc", "test_acc", "notes"])
    df.to_csv(os.path.join(out_dir, "experiment_summary.csv"), index=False)
    print("\nExperiment summary saved to", os.path.join(out_dir, "experiment_summary.csv"))
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
