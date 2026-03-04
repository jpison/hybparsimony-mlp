# install once:
#   pip install pydl8.5 pymurtree

import numpy as np, pandas as pd, os, pickle, warnings, concurrent.futures as cf
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.metrics import log_loss
from dl85 import DL85Classifier                    # DL8.5
from pymurtree import OptimalDecisionTreeClassifier  # MurTree
from scores import interpretability_score, set_tau, get_tau

MAX_DEPTH   = 4
N_RUNS      = 30
TAU_DEFAULT = 0.80
DATA_FOLDER = "./datasets/"
DATA_LIST   = pd.read_csv("datasets.csv")['name_file']

def fit_and_score(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    preds = clf.predict_proba(Xte)[:, 1]
    cv    = -cross_val_score(clf, Xtr, ytr,
                             cv=RepeatedKFold(5, 10, random_state=1234),
                             scoring="neg_log_loss", n_jobs=1).mean()
    tst   = log_loss(yte, preds)
    tree  = clf.get_tree() if hasattr(clf, "get_tree") else clf  # DL85 exposes get_tree()
    depth = tree.depth
    leaves= tree.n_leaves
    usedf = np.unique(tree.features).size
    nfs   = Xtr.shape[1]
    igeo  = interpretability_score(tree, usedf,
                                   Xtr, ytr, tau=get_tau())
    return cv, tst, igeo, depth, leaves, usedf

def experiment(run, filename):
    X = pd.read_csv(os.path.join(DATA_FOLDER, filename))
    y = X.pop('target_end')
    Xtr,Xte,ytr,yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1234*run)

    # DL8.5
    dl85 = DL85Classifier(max_depth=MAX_DEPTH, min_sup=1, time_limit=60)
    dl_cv, dl_tst, dl_geo, dl_d, dl_l, dl_f = fit_and_score(
        dl85, Xtr, ytr, Xte, yte)

    # MurTree
    mtree = OptimalDecisionTreeClassifier(max_depth=MAX_DEPTH,
                                          max_num_nodes=None, time=60)
    mt_cv, mt_tst, mt_geo, mt_d, mt_l, mt_f = fit_and_score(
        mtree, Xtr, ytr, Xte, yte)

    return {
        "dataset": filename, "run": run,
        "dl85_cv": dl_cv, "dl85_tst": dl_tst, "dl85_geo": dl_geo,
        "dl85_d": dl_d, "dl85_l": dl_l, "dl85_f": dl_f,
        "mt_cv": mt_cv, "mt_tst": mt_tst, "mt_geo": mt_geo,
        "mt_d": mt_d, "mt_l": mt_l, "mt_f": mt_f,
    }

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_tau(TAU_DEFAULT)

    out_rows = []
    with cf.ProcessPoolExecutor() as ex:
        futures = [ex.submit(experiment, r, fn)
                   for r in range(N_RUNS) for fn in DATA_LIST]
        for f in cf.as_completed(futures):
            out_rows.append(f.result())
            pd.DataFrame(out_rows).to_csv("results_baselines.csv", index=False)
