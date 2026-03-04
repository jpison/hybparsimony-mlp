import os

# Configuracion de paralelismo: 10 cores por proceso worker
CORES_PER_WORKER = 10
TOTAL_CORES = os.cpu_count() or 1
MAX_WORKERS = max(1, TOTAL_CORES // CORES_PER_WORKER)

# Limita hilos internos de BLAS/OpenMP por proceso worker
os.environ["OMP_NUM_THREADS"] = str(CORES_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(CORES_PER_WORKER)
os.environ["OPENBLAS_NUM_THREADS"] = str(CORES_PER_WORKER)
os.environ["NUMEXPR_NUM_THREADS"] = str(CORES_PER_WORKER)
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"

import numpy as np
import pandas as pd
import pickle
import json
import warnings
from functools import partial

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from hybparsimony import HYBparsimony, Population
from hybparsimony.util.complexity import mlp_complexity
from mlp_rules import extract_mlp_rules, format_rules_text

from scores import *


# ----------------------- Carpeta de modelos (sin limpieza automática) -----------------------
CARPETA_MODELOS = "mejores_modelos"
os.makedirs(CARPETA_MODELOS, exist_ok=True)

# ----------------------- Preparación de datos y CSV -----------------------
warnings.filterwarnings("ignore")

FOLDER_DATASETS = './datasets/'
df = pd.read_csv("datasets_ordenado.csv")
df = df[df['name_file'].apply(lambda fn: os.path.exists(FOLDER_DATASETS + fn))].reset_index(drop=True)


# --------------------- Parametros principales --------------
VERSION = 8
MAX_TOTAL_DEPTH = 4  # profundidad máxima permitida
NUM_RUNS = 25
MAX_ITERS = 75 #200
CV_N_SPLITS = 5
CV_N_REPEATS = 4
TAU_FINAL = 0.90
# Completados: 0-4, 
lista_datasets = df['name_file']    #[0:4] 

# Resumen inicial de datasets a tratar: pos, dataset, nrows, ncols
selected_names = [str(x) for x in list(lista_datasets)]
ncols_map = {}
if os.path.exists("datasets_nrows_ncols.csv"):
    dims_df = pd.read_csv("datasets_nrows_ncols.csv")[["dataset", "ncols"]]
    ncols_map = dict(zip(dims_df["dataset"].astype(str), pd.to_numeric(dims_df["ncols"], errors="coerce")))

rows_resumen = []
for ds_name in selected_names:
    matches = df.index[df["name_file"].astype(str) == ds_name].tolist()
    if not matches:
        rows_resumen.append({"pos": np.nan, "dataset": ds_name, "nrows": np.nan, "ncols": np.nan})
        continue

    pos = int(matches[0])
    row = df.loc[pos]
    nrows_val = pd.to_numeric(row.get("nrows", np.nan), errors="coerce")
    ncols_val = ncols_map.get(ds_name, pd.to_numeric(row.get("NFs", np.nan), errors="coerce"))
    rows_resumen.append({"pos": pos, "dataset": ds_name, "nrows": nrows_val, "ncols": ncols_val})

resumen_ds = pd.DataFrame(rows_resumen)
print(resumen_ds[["pos", "dataset", "nrows", "ncols"]].to_string(index=False))

# Nombre de salida con timestamp
now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
OUTPUT_CSV = f'./resultados/resultados_binarios{VERSION}_{now}.csv'
OUTPUT_MEDIA_CSV = f'./resultados/resultados_media_binarios{VERSION}_{now}.csv'
LOCK_FILE = OUTPUT_CSV + '.lock'

# Cabecera del CSV
columns = [
    'dataset','nrows','ncols',
    'run','seed',
    'i_5CV_logloss','p_5CV_logloss',
    'i_TST_logloss','p_TST_logloss',
    'i_geo_complexity','p_geo_complexity',
    'i_nfs_complexity','p_nfs_complexity',
    'i_H_total', 'i_H_active', 'i_C_inconsistentes', 'i_k_promedio_activas',
    'p_H_total', 'p_H_active', 'p_C_inconsistentes', 'p_k_promedio_activas',
    'i_best_model','p_best_model',
    
    'i_d', 'p_d',
    'i_l', 'p_l',
    'i_f', 'p_f',
    'i_tree_nfs', 'p_tree_nfs',

    'i_tree_used_fs', 'p_tree_used_fs',
    'i_selected_fs', 'p_selected_fs'
]

INTERPRETABLE_MODEL = "mlp"  # "mlp" o "tree"
# Parámetros de extracción de reglas MLP
RULES_DELTA = 1e-3
RULES_SCORE_QUANTILE = 0.25
RULES_EPS_R = 0.1
RULES_EPS_A = 0.1


def get_used_input_features_mlp(model, selected_features, delta=1e-3):
    """Devuelve nombres de features con al menos una conexión activa hacia capa oculta."""
    if not hasattr(model, "coefs_") or len(model.coefs_) < 1:
        return np.array([], dtype=object)
    w_in = model.coefs_[0]  # shape (d, H)
    selected_features = np.asarray(selected_features)
    used_idx = np.where(np.any(np.abs(w_in) > delta, axis=1))[0]
    return selected_features[used_idx]

def transform_for_model_input(model, X_block):
    """
    Transforma X con el scaler del modelo si existe.
    Para modelos sin scaler asociado, devuelve X como ndarray.
    """
    X_arr = X_block.values if "pandas" in str(type(X_block)) else np.asarray(X_block)
    scaler = getattr(model, "_input_scaler", None)
    if scaler is None:
        return X_arr
    return scaler.transform(X_arr)

def save_mlp_rules_artifacts(model, X_ref, selected_features, dataset_name, run, prefix):
    """
    Guarda reglas MLP en TXT y JSON para un best_model.
    prefix: 'i' o 'p'.
    """
    # Solo aplica a MLP de 1 capa oculta
    if not hasattr(model, "coefs_") or len(model.coefs_) != 2:
        return None

    features_arr = np.asarray(selected_features)
    pack = extract_mlp_rules(
        model=model,
        X=X_ref,
        feature_names=features_arr,
        delta=RULES_DELTA,
        score_quantile=RULES_SCORE_QUANTILE,
        eps_r=RULES_EPS_R,
        eps_a=RULES_EPS_A,
    )

    base_name = f"{prefix}_reglas_{os.path.splitext(dataset_name)[0]}_run_{run:02d}"
    txt_path = os.path.join(CARPETA_MODELOS, f"{base_name}.txt")
    json_path = os.path.join(CARPETA_MODELOS, f"{base_name}.json")

    with open(txt_path, "w", encoding="utf-8") as ftxt:
        ftxt.write(format_rules_text(pack))
    with open(json_path, "w", encoding="utf-8") as fjson:
        json.dump(pack, fjson, ensure_ascii=False, indent=2)

    return pack.get("summary", None)


def build_interpretable_mlp_cfg():
    """
    Configuración lista para HYBPARSIMONY-IMLP (1 capa oculta).
    Usa la métrica interpretability_score_mlp definida en scores.py.
    """
    return {
        "estimator": MLPClassifier,
        "complexity": interpretability_score_mlp,
        "hidden_layer_sizes": {"range": (1, 15), "type": Population.INTEGER},
        "alpha": {"range": (1e-5, 1e4), "type": Population.FLOAT},
        "activation": {"value": "logistic", "type": Population.CONSTANT},
        "solver": {"value": "lbfgs", "type": Population.CONSTANT},
        "max_iter": {"value": 500, "type": Population.CONSTANT},
        "random_state": {"value": 12345, "type": Population.CONSTANT},
    }

def build_parsimonious_mlp_cfg():
    """
    Modelo parsimonioso MLP usando la complejidad por defecto de HYBPARSIMONY
    para MLP: mlp_complexity.
    """
    return {
        "estimator": MLPClassifier,
        "complexity": mlp_complexity,
        "hidden_layer_sizes": {"range": (1, 15), "type": Population.INTEGER},
        "alpha": {"range": (1e-5, 1e4), "type": Population.FLOAT},
        "activation": {"value": "logistic", "type": Population.CONSTANT},
        "solver": {"value": "lbfgs", "type": Population.CONSTANT},
        "max_iter": {"value": 500, "type": Population.CONSTANT},
        "random_state": {"value": 12345, "type": Population.CONSTANT},
    }


# ----------------------- Funcion de un experimento -----------------------

def experimento(params):
    TAU = get_tau()
    run, name_file = params
    # Carga y partición
    X = pd.read_csv(FOLDER_DATASETS + name_file)
    nrows = X.shape[0]
    ncols = X.shape[1]

    y = X.pop('target_end')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1234 * run
    )
    total_feats = X.shape[1]
    input_names = X.columns

    # Modelo interpretable
    # -------------------->
    if INTERPRETABLE_MODEL == "mlp":
        interpretable_cfg = build_interpretable_mlp_cfg()
    else:
        interpretable_cfg = {
            "estimator": DecisionTreeClassifier,
            "complexity": interpretability_score_weighted,
            "criterion": {"value": "gini", "type": Population.CONSTANT},
            "class_weight": {"value": "balanced", "type": Population.CONSTANT},
            "splitter": {"value": "best", "type": Population.CONSTANT},
            "max_depth": {"range": (1, MAX_TOTAL_DEPTH), "type": Population.INTEGER},
            "min_samples_split": {"range": (2, int(np.ceil(X_train.shape[0] * 0.20))), "type": Population.INTEGER},
            "min_samples_leaf": {"range": (1, int(np.ceil(X_train.shape[0] * 0.20))), "type": Population.INTEGER},
            "max_features": {"value": None, "type": Population.CONSTANT},
            "random_state": {"value": 1234, "type": Population.CONSTANT},
            "ccp_alpha": {"range": (0, 1), "type": Population.FLOAT},
        }

    


    model_i = HYBparsimony(
        fitness=getFitness_custom(interpretable_cfg['estimator'], 
                           interpretable_cfg['complexity'], 
                           partial(default_cv_score, n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)),
        algorithm=interpretable_cfg,
        features=input_names,
        #cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=1234),
        n_jobs=1,
        maxiter=MAX_ITERS,
        rerank_error=0.01,
        npart=30,
        keep_history=True,
        seed_ini=1234 * run,
        verbose=0
    )
    model_i.fit(X_train, y_train)
    i_rules_summary = None

    # Predicciones y métricas (5-CV y test)
    try:
        X_test_i = transform_for_model_input(model_i.best_model, X_test[model_i.selected_features])
        X_train_i = transform_for_model_input(model_i.best_model, X_train[model_i.selected_features])

        preds_i = model_i.best_model.predict_proba(X_test_i)[:, 1]
        i_5CV    = -round(model_i.best_score, 6)
        i_TST    = round(log_loss(y_test, preds_i), 6)
        nfs_i    = len(model_i.selected_features)
        if INTERPRETABLE_MODEL == "mlp":
            i_geo = interpretability_score_mlp(
                model_i.best_model,
                nfs_i,
                X_train_i,
                y_train,
                d_total_dataset=total_feats,
                tau=TAU,
            )
            i_nfs_c = nfs_i
        else:
            i_geo = interpretability_score_weighted(
                model_i.best_model,
                nfs_i,
                X_train_i,
                y_train,
                TAU,
            )
            i_nfs_c = decision_tree_complexity(model_i.best_model, nfs_i) // 1e9
    except:
        i_5CV = 99.9
        i_TST = 99.9
        nfs_i = 99.9
        i_geo = 99.9
        i_nfs_c = 99.9



    # Complejidades
    
    i_best   = str(model_i.best_model)
    i_feats  = model_i.selected_features
    if INTERPRETABLE_MODEL == "mlp":
        i_d = np.nan
        i_l = np.nan
        i_f = np.nan
        i_tree_used_feature_names = get_used_input_features_mlp(model_i.best_model, model_i.selected_features)
    else:
        # Sacar datos del árbol para i_best_model
        tree_i   = model_i.best_model
        i_d      = tree_i.get_depth()
        i_l      = tree_i.get_n_leaves()
        i_f      = np.unique(tree_i.tree_.feature[tree_i.tree_.feature >= 0]).size
        # Obtener las variables usadas en el árbol
        used_feature_indices = tree_i.tree_.feature
        used_feature_indices = used_feature_indices[used_feature_indices >= 0]
        i_tree_used_feature_names = model_i.selected_features[np.unique(used_feature_indices)]


    # Guardar modelo interpretable
    dump_i = {
        "run": run, "dataset": name_file,
        "history": model_i.history,
        "best_model": model_i.best_model,
        "selected_features": i_feats,
        "tree_features": i_tree_used_feature_names
    }
    with open(f"mejores_modelos/i_{os.path.splitext(name_file)[0]}_run_{run:02d}.pkl", "wb") as f:
        pickle.dump(dump_i, f)
    # Guardar reglas del mejor MLP interpretable
    try:
        i_rules_summary = save_mlp_rules_artifacts(
            model=model_i.best_model,
            X_ref=transform_for_model_input(model_i.best_model, X_train[model_i.selected_features]),
            selected_features=model_i.selected_features,
            dataset_name=name_file,
            run=run,
            prefix="i",
        )
    except Exception as e:
        print(f"[WARN] No se pudieron guardar reglas i_MLP para {name_file} run {run}: {e}")
    i_H_total = np.nan
    i_H_active = np.nan
    i_C_inconsistentes = np.nan
    i_k_promedio_activas = np.nan
    if i_rules_summary is not None:
        i_H_total = i_rules_summary.get("H_total", np.nan)
        i_H_active = i_rules_summary.get("H_active", np.nan)
        i_comp = i_rules_summary.get("complexity_components", {})
        i_C_inconsistentes = i_comp.get("C_inconsistentes", np.nan)
        i_k_promedio_activas = i_comp.get("k_promedio_activas", np.nan)

    # Modelo parsimonioso (MLP + complejidad por defecto de HYBPARSIMONY para MLP)
    # -----------------------------------------------------------------------------
    parsimonious_cfg = build_parsimonious_mlp_cfg()
    model_p = HYBparsimony(
        fitness=getFitness_custom(parsimonious_cfg['estimator'],
                           parsimonious_cfg['complexity'],
                           partial(default_cv_score, n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS)),
        algorithm=parsimonious_cfg,
        features=input_names,
        n_jobs=1,
        maxiter=MAX_ITERS,
        rerank_error=0.01,
        npart=30,
        keep_history=True,
        seed_ini=1234 * run,
        verbose=0
    )
    model_p.fit(X_train, y_train)
    p_rules_summary = None

    # Predicciones y métricas (5-CV y test)
    try:
        X_test_p = transform_for_model_input(model_p.best_model, X_test[model_p.selected_features])
        X_train_p = transform_for_model_input(model_p.best_model, X_train[model_p.selected_features])

        preds_p  = model_p.best_model.predict_proba(X_test_p)[:, 1]
        p_5CV    = -round(model_p.best_score, 6)
        p_TST    = round(log_loss(y_test, preds_p), 6)
        # Complejidades
        nfs_p    = len(model_p.selected_features)
        p_geo    = interpretability_score_mlp(
            model_p.best_model,
            nfs_p,
            X_train_p,
            y_train,
            d_total_dataset=total_feats,
            tau=TAU,
        )
        p_nfs_c  = mlp_complexity(model_p.best_model, nfs_p) // 1e9
    except:
        p_5CV = 99.9
        p_TST = 99.9
        nfs_p = 99.9
        p_geo = 99.9
        p_nfs_c = 99.9

    p_best   = str(model_p.best_model)
    p_feats  = model_p.selected_features
    p_d = np.nan
    p_l = np.nan
    p_f = np.nan
    p_tree_used_feature_names = get_used_input_features_mlp(model_p.best_model, model_p.selected_features)

    # Guardar modelo parsimonioso
    dump_p = {
        "run": run, "dataset": name_file, 
        "history": model_p.history,
        "best_model": model_p.best_model,
        "selected_features": p_feats,
        "tree_features": p_tree_used_feature_names
    }
    with open(f"mejores_modelos/p_{os.path.splitext(name_file)[0]}_run_{run:02d}.pkl", "wb") as f:
        pickle.dump(dump_p, f)
    # Guardar reglas del mejor MLP parsimonioso
    try:
        p_rules_summary = save_mlp_rules_artifacts(
            model=model_p.best_model,
            X_ref=transform_for_model_input(model_p.best_model, X_train[model_p.selected_features]),
            selected_features=model_p.selected_features,
            dataset_name=name_file,
            run=run,
            prefix="p",
        )
    except Exception as e:
        print(f"[WARN] No se pudieron guardar reglas p_MLP para {name_file} run {run}: {e}")
    p_H_total = np.nan
    p_H_active = np.nan
    p_C_inconsistentes = np.nan
    p_k_promedio_activas = np.nan
    if p_rules_summary is not None:
        p_H_total = p_rules_summary.get("H_total", np.nan)
        p_H_active = p_rules_summary.get("H_active", np.nan)
        p_comp = p_rules_summary.get("complexity_components", {})
        p_C_inconsistentes = p_comp.get("C_inconsistentes", np.nan)
        p_k_promedio_activas = p_comp.get("k_promedio_activas", np.nan)

    return {
        'dataset': name_file, 'run': run, 'seed': 1234 * run,
        'nrows':nrows, 'ncols':ncols,
        'i_5CV_logloss': i_5CV, 'p_5CV_logloss': p_5CV,
        'i_TST_logloss': i_TST, 'p_TST_logloss': p_TST,
        'i_geo_complexity': i_geo, 'p_geo_complexity': p_geo,
        'i_nfs_complexity': i_nfs_c, 'p_nfs_complexity': p_nfs_c,
        'i_H_total': i_H_total, 'i_H_active': i_H_active,
        'i_C_inconsistentes': i_C_inconsistentes, 'i_k_promedio_activas': i_k_promedio_activas,
        'p_H_total': p_H_total, 'p_H_active': p_H_active,
        'p_C_inconsistentes': p_C_inconsistentes, 'p_k_promedio_activas': p_k_promedio_activas,
        'i_best_model': i_best, 'p_best_model': p_best,
        
        'i_d': i_d, 'i_l': i_l, 'i_f': i_f,
        'p_d': p_d, 'p_l': p_l, 'p_f': p_f,

        'i_tree_nfs': len(i_tree_used_feature_names), 'p_tree_nfs': len(p_tree_used_feature_names),
        'i_tree_used_fs': i_tree_used_feature_names, 'p_tree_used_fs': p_tree_used_feature_names,
        'i_selected_fs': str(i_feats), 'p_selected_fs': str(p_feats)
    }

# ----------------------- Ejecucion en paralelo -----------------------
import sys
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from filelock import FileLock
from pandas import DataFrame


# df debe estar ya cargado arriba, p.e. con pandas.read_csv(...)
tasks = [(run, fn) for run in range(NUM_RUNS) for fn in lista_datasets]


def main():
    set_tau(TAU_FINAL)
    
     # Asume que 'experimento', 'tasks', 'OUTPUT_CSV', 'LOCK_FILE' y 'columns' están definidos en el módulo
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

    # Handler para Ctrl+C: cierra el executor y sale
    def handler(signum, frame):
        print("\nRecibido SIGINT. Cerrando workers")
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    resultados = []
    # Lanzar todas las tareas en paralelo
    futures = [executor.submit(experimento, t) for t in tasks]

    try:
        # tqdm sobre los futures completados
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Procesos completados",
            unit="tarea"
        ):
            res = fut.result()
            resultados.append(res)
            # Guarda resultados
            df_tmp = pd.DataFrame(resultados, columns=columns)
            df_tmp.to_csv(OUTPUT_CSV, index=False)
            
            # Calcular la media por dataset para las columnas restantes
            columnas_incluir = ['run', 'seed', 'nrows', 'ncols',
                                'i_5CV_logloss','p_5CV_logloss',
                                'i_TST_logloss','p_TST_logloss',
                                'i_geo_complexity','p_geo_complexity',
                                'i_nfs_complexity','p_nfs_complexity',
                                'i_H_total', 'i_H_active', 'i_C_inconsistentes', 'i_k_promedio_activas',
                                'p_H_total', 'p_H_active', 'p_C_inconsistentes', 'p_k_promedio_activas',
                                # 'i_best_model','p_best_model',
                                
                                'i_d', 'p_d',
                                'i_l', 'p_l',
                                'i_f', 'p_f',
                                'i_tree_nfs', 'p_tree_nfs',

                                # 'i_tree_used_fs', 'p_tree_used_fs',
                                # 'i_selected_fs', 'p_selected_fs'
                                ]
            try:
                df_tmp = df_tmp.query('i_5CV_logloss<99.0 & p_5CV_logloss<99.0').reset_index(drop=True)
                media_por_dataset = df_tmp.groupby('dataset')[columnas_incluir].mean().reset_index()
                media_redondeada = media_por_dataset.round(4)
                cols_new_metrics = [
                    'i_H_total', 'i_H_active', 'i_C_inconsistentes', 'i_k_promedio_activas',
                    'p_H_total', 'p_H_active', 'p_C_inconsistentes', 'p_k_promedio_activas'
                ]
                media_redondeada[cols_new_metrics] = media_redondeada[cols_new_metrics].round(2)
                media_redondeada.to_csv(OUTPUT_MEDIA_CSV, index=False)
            except:
                print('No se ha podido hacer la media')


    except KeyboardInterrupt:
        print("\nInterrupción por teclado. Cerrando executor")
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        # Asegurarse de liberar recursos
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
