#!/usr/bin/env python
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning


def resolve_dataset_path(arg):
    if os.path.exists(arg):
        return arg
    base = arg
    if not base.lower().endswith('.csv'):
        base = base + '.csv'
    candidate = os.path.join('datasets', base)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"No se encontro el dataset: {arg}")


def encode_binary_target(y):
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Se esperaba clasificacion binaria, clases encontradas: {classes}")
    if set(classes) == {0, 1}:
        return y.astype(int)
    mapping = {classes[0]: 0, classes[1]: 1}
    return y.map(mapping).astype(int)


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def i_mlp(model, X_val, delta=1e-3, epsilon=0.1, epsilon_out=0.1):
    W_in = model.coefs_[0]          # (d, H)
    W_out = model.coefs_[1].ravel() # (H,)
    b_hidden = model.intercepts_[0]

    H = W_in.shape[1]
    if H == 0:
        return 0.0

    # 1) Sparsidad por neurona
    active_weights = np.abs(W_in) > delta
    k_per_neuron = active_weights.sum(axis=0)
    avg_k = k_per_neuron.mean()

    # 2) Consistencia de signos
    P = W_in * W_out  # broadcast
    sign_matrix = np.sign(P) * active_weights
    inconsistent = 0
    for i in range(sign_matrix.shape[0]):
        signs_i = sign_matrix[i]
        signs_i = signs_i[signs_i != 0]
        if signs_i.size == 0:
            continue
        if (signs_i > 0).any() and (signs_i < 0).any():
            inconsistent += 1

    # 3) Rango de activacion
    Xv = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    Z = Xv @ W_in + b_hidden
    Hval = sigmoid(Z)
    r = Hval.max(axis=0) - Hval.min(axis=0)

    # 4) Neuronas muertas
    a = np.abs(W_out) * r
    dead = (r < epsilon) | (a < epsilon_out)
    H_dead = int(dead.sum())

    # 5) Metricas finales
    score = (H * 1e6) + (inconsistent * 1e3) + (avg_k * 1e2) + (H_dead * 1e2)
    return float(score)


def evaluate_config(X, y, splits, n_hidden, alpha):
    loglosses = []
    scores = []

    for train_idx, val_idx in splits:
        X_train_df = X.iloc[train_idx]
        X_val_df = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Imputacion simple con mediana (por fold) si hay NaNs
        if X_train_df.isnull().any().any():
            med = X_train_df.median()
            X_train_df = X_train_df.fillna(med)
            X_val_df = X_val_df.fillna(med)

        X_train = X_train_df.to_numpy(dtype=float)
        X_val = X_val_df.to_numpy(dtype=float)

        model = MLPClassifier(
            solver='lbfgs',
            random_state=12345,
            activation='logistic',
            hidden_layer_sizes=(int(n_hidden),),
            alpha=float(alpha),
            max_iter=500
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, probs, labels=[0, 1])
        im = i_mlp(model, X_val)

        loglosses.append(ll)
        scores.append(im)

    return float(np.mean(loglosses)), float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="Prueba de MLP con metrica i_mlp")
    parser.add_argument('base_datos', help='Ruta o nombre del CSV en datasets/')
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.base_datos)
    df = pd.read_csv(dataset_path)

    if 'target_end' in df.columns:
        y = df.pop('target_end')
    elif 'target' in df.columns:
        y = df.pop('target')
    else:
        raise ValueError("No se encontro columna objetivo ('target_end' o 'target').")

    # One-hot si hay categoricas
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df = pd.get_dummies(df, columns=non_numeric, drop_first=True)

    y = encode_binary_target(y)

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=12345)
    splits = list(cv.split(df, y))

    rng = np.random.default_rng(12345)
    n_list = rng.integers(1, 21, size=10)
    alphas = 10 ** rng.uniform(-5, 4, size=10)

    results = []

    print(f"Dataset: {dataset_path}")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("RepeatedKFold: 5 folds x 10 repeticiones = 50 folds")
    print("Evaluando 10 MLPs...\n")

    for i, (n_hidden, alpha) in enumerate(zip(n_list, alphas), start=1):
        mean_ll, mean_imlp = evaluate_config(df, y, splits, n_hidden, alpha)
        results.append({
            'n': int(n_hidden),
            'alpha': float(alpha),
            'mean_logloss': mean_ll,
            'mean_i_mlp': mean_imlp
        })
        print(f"[{i:02d}] n={int(n_hidden):2d} | alpha={alpha:.3e} | mean_logloss={mean_ll:.6f} | mean_i_mlp={mean_imlp:.2f}")

    # Resumen en tabla
    print("\nResumen:")
    res_df = pd.DataFrame(results)
    res_df = res_df[['n', 'alpha', 'mean_logloss', 'mean_i_mlp']]
    print(res_df.to_string(index=False, formatters={
        'alpha': lambda x: f"{x:.3e}",
        'mean_logloss': lambda x: f"{x:.6f}",
        'mean_i_mlp': lambda x: f"{x:.2f}"
    }))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
