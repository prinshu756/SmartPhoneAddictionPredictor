import itertools
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

RANDOM_STATE = 42
TARGET = "addicted_label"

train_df = pd.read_csv("datasets/train.csv")
y = train_df[TARGET].astype(int).values

names = ["LightGBM", "LightGBMDeep", "XGBoost", "HistGB"]
oofs = []
tests = []
for name in names:
    data = np.load(f"oof_{name}.npz")
    oofs.append(data["oof"])
    tests.append(data["test"])
    print(f"{name}: OOF AUC {data['auc']:.6f}")

oof_matrix = np.vstack(oofs)
test_matrix = np.vstack(tests)

print("\nEqual-weight blends:")
best_combo = None
for r in range(2, len(names) + 1):
    for combo in itertools.combinations(range(len(names)), r):
        blend = oof_matrix[list(combo)].mean(axis=0)
        auc = roc_auc_score(y, blend)
        label = " + ".join(names[i] for i in combo)
        print(f"  {label}: {auc:.6f}")
        if best_combo is None or auc > best_combo[1]:
            best_combo = (combo, auc, "equal")

print(f"\nBest equal blend: {best_combo[1]:.6f}")

n = len(names)


def neg_auc(weights):
    w = np.abs(weights)
    w = w / w.sum()
    return -roc_auc_score(y, (oof_matrix * w[:, None]).sum(axis=0))


results = []
for seed in range(2):
    x0 = np.random.RandomState(seed).dirichlet(np.ones(n))
    res = minimize(neg_auc, x0, method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-7})
    w = np.abs(res.x)
    w = w / w.sum()
    auc = roc_auc_score(y, (oof_matrix * w[:, None]).sum(axis=0))
    results.append((auc, w))

results.sort(key=lambda t: -t[0])
best_auc, best_w = results[0]
print("\nBest optimized blend OOF AUC: {:.6f}".format(best_auc))
for name, weight in zip(names, best_w):
    print(f"  {name}: {weight:.4f}")

final_test = (test_matrix * best_w[:, None]).sum(axis=0)
submission = pd.DataFrame(
    {"id": train_df["id"].iloc[-1] + 1 + np.arange(len(final_test)),
     "addicted_label": np.clip(final_test, 0, 1)}
)
np.savez("blend_result.npz", oof=(oof_matrix * best_w[:, None]).sum(axis=0),
         test=final_test, auc=best_auc, weights=best_w, names=np.array(names))
print("Saved blend_result.npz")
