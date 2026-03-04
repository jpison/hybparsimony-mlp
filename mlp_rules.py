import numpy as np


def _sigmoid_stable(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _as_2d_array(X):
    if "pandas" in str(type(X)):
        X = X.values
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X debe tener forma 2D (n_samples, n_features).")
    return X


def _validate_single_hidden_layer(model):
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise ValueError("El modelo no parece ser un MLP entrenado de sklearn.")
    if len(model.coefs_) != 2:
        raise ValueError("Solo se soporta MLP con una capa oculta.")


def _feature_names(feature_names, d):
    if feature_names is None:
        return np.array([f"x{i}" for i in range(d)], dtype=object)
    names = np.asarray(feature_names, dtype=object)
    if names.shape[0] != d:
        raise ValueError("feature_names no coincide con el numero de features.")
    return names


def _p_mlp_complexity(model, n_features):
    """
    Complejidad clasica tipo HYBPARSIMONY para MLP:
      p_mlp = n_features*1e9 + sum(weights^2) (acotado internamente a 1e9-1)
    """
    weights = [np.concatenate(model.intercepts_)]
    for wm in model.coefs_:
        weights.append(wm.flatten())
    weights = np.concatenate(weights)
    int_comp = np.min((1e9 - 1, np.sum(weights ** 2)))
    return float(n_features * 1e9 + int_comp)


def compute_dead_neurons(model, X, eps_r=0.1, eps_a=0.1):
    """
    Neurona muerta si:
      r_j < eps_r  o  |W_out_j| * r_j < eps_a
    """
    _validate_single_hidden_layer(model)
    Xv = _as_2d_array(X)

    W_in = model.coefs_[0]
    W_out = model.coefs_[1].ravel()
    b_hidden = model.intercepts_[0]

    if Xv.shape[1] != W_in.shape[0]:
        raise ValueError("X y modelo tienen distinto numero de features.")

    z = Xv @ W_in + b_hidden
    h = _sigmoid_stable(z)
    r = h.max(axis=0) - h.min(axis=0)
    a = np.abs(W_out) * r

    dead_mask = (r < eps_r) | (a < eps_a)
    active_mask = ~dead_mask

    return {
        "ranges": r,
        "amplitudes": a,
        "dead_mask": dead_mask,
        "active_mask": active_mask,
        "H_total": int(W_in.shape[1]),
        "H_dead": int(dead_mask.sum()),
        "H_active": int(active_mask.sum()),
    }


def compute_connection_scores(model, X, eps_r=0.1, eps_a=0.1):
    """
    Score de relevancia por conexion (i, j):
      score_ij = |W_in_ij| * |W_out_j| * r_j
    """
    _validate_single_hidden_layer(model)
    W_in = model.coefs_[0]
    W_out = model.coefs_[1].ravel()
    dead_info = compute_dead_neurons(model, X, eps_r=eps_r, eps_a=eps_a)
    r = dead_info["ranges"]

    per_neuron_gain = np.abs(W_out) * r
    score = np.abs(W_in) * per_neuron_gain[None, :]
    score[:, dead_info["dead_mask"]] = 0.0

    return score, dead_info


def prune_connections(
    model,
    X,
    delta=1e-3,
    score_threshold=None,
    score_quantile=0.25,
    eps_r=0.1,
    eps_a=0.1,
):
    """
    Conexion activa base: |W_in_ij| > delta
    Conexion relevante final: activa base y score_ij >= threshold
    """
    _validate_single_hidden_layer(model)
    W_in = model.coefs_[0]

    score, dead_info = compute_connection_scores(model, X, eps_r=eps_r, eps_a=eps_a)
    base_active = np.abs(W_in) > delta

    if score_threshold is None:
        candidates = score[base_active & (score > 0)]
        if candidates.size == 0:
            threshold = 0.0
        else:
            threshold = float(np.quantile(candidates, score_quantile))
    else:
        threshold = float(score_threshold)

    keep_mask = base_active & (score >= threshold)
    keep_mask[:, dead_info["dead_mask"]] = False

    return {
        "score": score,
        "base_active_mask": base_active,
        "keep_mask": keep_mask,
        "threshold": threshold,
        "dead_info": dead_info,
    }


def detect_inconsistent_features(model, keep_mask, feature_names=None):
    """
    Inconsistencia por feature usando signo efectivo sign(W_in_ij * W_out_j)
    sobre las conexiones mantenidas (keep_mask).
    """
    _validate_single_hidden_layer(model)
    W_in = model.coefs_[0]
    W_out = model.coefs_[1].ravel()
    d = W_in.shape[0]
    names = _feature_names(feature_names, d)

    report = []
    for i in range(d):
        mask_i = keep_mask[i, :]
        if not np.any(mask_i):
            report.append(
                {
                    "feature_index": int(i),
                    "feature_name": str(names[i]),
                    "used": False,
                    "inconsistent": False,
                    "pos_neurons": [],
                    "neg_neurons": [],
                }
            )
            continue

        contrib = W_in[i, mask_i] * W_out[mask_i]
        neuron_idx = np.where(mask_i)[0]
        pos_neurons = neuron_idx[contrib > 0]
        neg_neurons = neuron_idx[contrib < 0]
        inconsistent = (pos_neurons.size > 0) and (neg_neurons.size > 0)

        report.append(
            {
                "feature_index": int(i),
                "feature_name": str(names[i]),
                "used": True,
                "inconsistent": bool(inconsistent),
                "pos_neurons": [int(x) for x in pos_neurons.tolist()],
                "neg_neurons": [int(x) for x in neg_neurons.tolist()],
            }
        )
    return report


def extract_mlp_rules(
    model,
    X,
    feature_names=None,
    delta=1e-3,
    score_threshold=None,
    score_quantile=0.25,
    eps_r=0.1,
    eps_a=0.1,
    w_inconsistency=0.6,
    w_sparsity=0.4,
):
    """
    Extrae reglas de una MLP (1 capa oculta), podando conexiones irrelevantes y
    marcando features inconsistentes.
    """
    _validate_single_hidden_layer(model)
    Xv = _as_2d_array(X)
    W_in = model.coefs_[0]
    W_out = model.coefs_[1].ravel()
    b_hidden = model.intercepts_[0]
    b_out = float(model.intercepts_[1].ravel()[0])
    d, _ = W_in.shape
    names = _feature_names(feature_names, d)

    prune = prune_connections(
        model,
        Xv,
        delta=delta,
        score_threshold=score_threshold,
        score_quantile=score_quantile,
        eps_r=eps_r,
        eps_a=eps_a,
    )
    keep_mask = prune["keep_mask"]
    dead_info = prune["dead_info"]

    neuron_rules = []
    rule_ids = []
    for j in np.where(dead_info["active_mask"])[0]:
        idx = np.where(keep_mask[:, j])[0]
        if idx.size == 0:
            continue

        terms = [f"({W_in[i, j]:+.4f}*{names[i]})" for i in idx]
        cond = " + ".join(terms) + f" {b_hidden[j]:+.4f} > 0"

        pos_feats = [str(names[i]) for i in idx if W_in[i, j] > 0]
        neg_feats = [str(names[i]) for i in idx if W_in[i, j] < 0]
        rule_id = f"R{len(rule_ids) + 1}"
        rule_ids.append((rule_id, int(j)))

        neuron_rules.append(
            {
                "rule_id": rule_id,
                "neuron_index": int(j),
                "output_weight": float(W_out[j]),
                "hidden_bias": float(b_hidden[j]),
                "features": [str(names[i]) for i in idx.tolist()],
                "positive_features": pos_feats,
                "negative_features": neg_feats,
                "condition": cond,
            }
        )

    output_terms = [
        f"({W_out[j]:+.4f}*I({rid}_ON))"
        for rid, j in rule_ids
    ]
    output_expr = f"{b_out:+.4f}"
    if output_terms:
        output_expr += " + " + " + ".join(output_terms)
    output_expr += " > 0"

    feature_report = detect_inconsistent_features(model, keep_mask, feature_names=names)
    inconsistent_features = [f for f in feature_report if f["inconsistent"]]

    # Complejidad i_mlp usada en el proyecto:
    # i_mlp = H_activas * (1 + 0.6*(C_inconsistentes/d) + 0.4*(k_promedio_activas/d))
    # Nota: k_promedio_activas se calcula sobre conexiones mantenidas en keep_mask.
    active_cols = np.where(dead_info["active_mask"])[0]
    if active_cols.size > 0:
        k_per_active_neuron = keep_mask[:, active_cols].sum(axis=0).astype(float)
        k_avg_active = float(k_per_active_neuron.mean())
    else:
        k_avg_active = 0.0

    c_inconsistent = int(len(inconsistent_features))
    d_safe = float(max(1, d))
    weight_sum = w_inconsistency + w_sparsity
    if weight_sum <= 0:
        w_inc = 0.6
        w_spa = 0.4
    else:
        w_inc = w_inconsistency / weight_sum
        w_spa = w_sparsity / weight_sum

    i_mlp = float(
        dead_info["H_active"]
        * (1.0 + w_inc * (c_inconsistent / d_safe) + w_spa * (k_avg_active / d_safe))
    )
    p_mlp = _p_mlp_complexity(model, d)

    return {
        "summary": {
            "n_samples": int(Xv.shape[0]),
            "d_features": int(d),
            "H_total": dead_info["H_total"],
            "H_active": dead_info["H_active"],
            "H_dead": dead_info["H_dead"],
            "kept_connections": int(keep_mask.sum()),
            "connection_threshold": float(prune["threshold"]),
            "n_inconsistent_features": int(len(inconsistent_features)),
            "complexity_metric_name": "i_mlp",
            "complexity_formula": "i_mlp = H_activas * (1 + 0.6*(C_inconsistentes/d) + 0.4*(k_promedio_activas/d))",
            "complexity_value": i_mlp,
            "p_mlp_value": p_mlp,
            "complexity_components": {
                "H_activas": int(dead_info["H_active"]),
                "C_inconsistentes": c_inconsistent,
                "d": int(d),
                "k_promedio_activas": k_avg_active,
                "peso_inconsistencias": float(w_inc),
                "peso_sparsidad": float(w_spa),
            },
        },
        "neuron_rules": neuron_rules,
        "output_rule": {
            "condition": output_expr,
            "meaning": "Clase positiva si la condicion se cumple; negativa en caso contrario.",
        },
        "feature_report": feature_report,
        "inconsistent_features": inconsistent_features,
    }


def format_rules_text(rule_pack):
    """
    Render simple en texto para inspeccion humana.
    """
    s = rule_pack["summary"]
    lines = [
        "=== Resumen ===",
        f"H_total={s['H_total']} | H_active={s['H_active']} | H_dead={s['H_dead']}",
        f"d={s['d_features']} | conexiones_mantenidas={s['kept_connections']} | threshold={s['connection_threshold']:.6f}",
        f"features_inconsistentes={s['n_inconsistent_features']}",
        f"{s['complexity_metric_name']}={s['complexity_value']:.6f}",
        f"p_mlp={s['p_mlp_value']:.6f}",
        "",
        "=== Complejidad ===",
        s["complexity_formula"],
        (
            "Sustituyendo: "
            f"H_activas={s['complexity_components']['H_activas']}, "
            f"C_inconsistentes={s['complexity_components']['C_inconsistentes']}, "
            f"d={s['complexity_components']['d']}, "
            f"k_promedio_activas={s['complexity_components']['k_promedio_activas']:.6f}, "
            f"peso_inconsistencias={s['complexity_components']['peso_inconsistencias']:.3f}, "
            f"peso_sparsidad={s['complexity_components']['peso_sparsidad']:.3f}"
        ),
        "",
        "=== Reglas por neurona oculta ===",
    ]

    if not rule_pack["neuron_rules"]:
        lines.append("No hay reglas activas tras la poda.")
    else:
        for r in rule_pack["neuron_rules"]:
            lines.append(
                f"{r['rule_id']} (neurona {r['neuron_index']}, W_out={r['output_weight']:+.4f}): "
                f"SI {r['condition']} ENTONCES {r['rule_id']}_ON"
            )

    lines.extend(
        [
            "",
            "=== Regla de salida ===",
            f"SI {rule_pack['output_rule']['condition']} ENTONCES clase=1; SI NO clase=0",
            "",
            "=== Features inconsistentes ===",
        ]
    )
    if not rule_pack["inconsistent_features"]:
        lines.append("Ninguna.")
    else:
        for f in rule_pack["inconsistent_features"]:
            lines.append(
                f"{f['feature_name']}: + en neuronas {f['pos_neurons']} | - en neuronas {f['neg_neurons']}"
            )

    return "\n".join(lines)
