import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from lightgbm import early_stopping as lgb_early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "addicted_label"
ID_COL = "id"

train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")


def create_features(df):
    df = df.copy()
    eps = 1e-6

    df["known_usage_hours"] = (
        df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
    )
    df["unaccounted_screen_time"] = df["daily_screen_time_hours"] - df["known_usage_hours"]
    df["screen_sleep_ratio"] = df["daily_screen_time_hours"] / (df["sleep_hours"] + eps)
    df["social_media_ratio"] = df["social_media_hours"] / (df["daily_screen_time_hours"] + eps)
    df["gaming_ratio"] = df["gaming_hours"] / (df["daily_screen_time_hours"] + eps)
    df["work_study_ratio"] = df["work_study_hours"] / (df["daily_screen_time_hours"] + eps)
    df["weekend_screen_difference"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    df["weekend_screen_ratio"] = df["weekend_screen_time"] / (df["daily_screen_time_hours"] + eps)
    df["notifications_per_screen_hour"] = df["notifications_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["app_opens_per_screen_hour"] = df["app_opens_per_day"] / (df["daily_screen_time_hours"] + eps)
    df["notifications_per_app_open"] = df["notifications_per_day"] / (df["app_opens_per_day"] + eps)
    df["sleep_deficit_8h"] = 8 - df["sleep_hours"]
    df["short_sleep_flag"] = (df["sleep_hours"] < 7).astype(int)
    df["high_screen_time_flag"] = (df["daily_screen_time_hours"] >= 8).astype(int)
    df["high_notification_flag"] = (df["notifications_per_day"] >= 100).astype(int)
    df["screen_time_x_social_media"] = df["daily_screen_time_hours"] * df["social_media_hours"]
    df["screen_time_x_app_opens"] = df["daily_screen_time_hours"] * df["app_opens_per_day"]

    # New: missing indicators
    for col in df.columns:
        if col not in (ID_COL, TARGET):
            df[col + "_missing"] = df[col].isna().astype(int)

    # New: total engagement intensity
    df["notifications_per_known_usage"] = df["notifications_per_day"] / (df["known_usage_hours"] + eps)
    df["app_opens_per_known_usage"] = df["app_opens_per_day"] / (df["known_usage_hours"] + eps)
    df["avg_session_length"] = df["daily_screen_time_hours"] / (df["app_opens_per_day"] + eps)
    df["total_screen_week"] = df["daily_screen_time_hours"] * 5 + df["weekend_screen_time"] * 2

    return df


train_features_df = create_features(train_df)
test_features_df = create_features(test_df)

X = train_features_df.drop(columns=[TARGET, ID_COL], errors="ignore")
y = train_features_df[TARGET].astype(int)
X_test = test_features_df.drop(columns=[ID_COL], errors="ignore")

print("X shape:", X.shape, "X_test shape:", X_test.shape)

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_columns = [c for c in X.columns if c not in categorical_columns]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ),
                ]
            ),
            categorical_columns,
        ),
    ],
    remainder="drop",
)

X_enc = tree_preprocessor.fit_transform(X)
X_test_enc = tree_preprocessor.transform(X_test)


def train_with_cv(model_builder, X_data, y_data, X_test_data, model_name, use_cat_features=False, use_eval_set=True):
    oof = np.zeros(len(X_data))
    test_pred = np.zeros(len(X_test_data))
    fold_scores = []
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_data, y_data), start=1):
        if isinstance(X_data, pd.DataFrame):
            X_tr = X_data.iloc[tr_idx]
            X_va = X_data.iloc[va_idx]
        else:
            X_tr = X_data[tr_idx]
            X_va = X_data[va_idx]
        y_tr = y_data.iloc[tr_idx]
        y_va = y_data.iloc[va_idx]

        model = model_builder(fold)
        if use_cat_features:
            model.fit(
                X_tr, y_tr,
                cat_features=categorical_columns,
                eval_set=(X_va, y_va),
                early_stopping_rounds=200,
                verbose=False,
            )
        elif use_eval_set:
            if "LGBM" in model_name or "LightGBM" in model_name:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=[lgb_early_stopping(200, verbose=False)],
                )
            else:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )
        else:
            model.fit(X_tr, y_tr)

        va_pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_pred += model.predict_proba(X_test_data)[:, 1] / N_SPLITS
        auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(auc)
        print(f"{model_name} | Fold {fold} | AUC: {auc:.6f}")

    overall = roc_auc_score(y_data, oof)
    print("=" * 60)
    print(f"{model_name} OOF ROC-AUC: {overall:.6f}")
    print(f"Mean Fold AUC: {np.mean(fold_scores):.6f}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("=" * 60)
    np.savez(
        f"oof_{model_name}.npz",
        oof=oof,
        test=test_pred,
        auc=overall,
        fold_scores=np.array(fold_scores),
    )
    return {"name": model_name, "oof": oof, "test": test_pred, "auc": overall}


def build_lightgbm(fold):
    return LGBMClassifier(
        objective="binary",
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1,
    )


def build_lightgbm_deep(fold):
    return LGBMClassifier(
        objective="binary",
        n_estimators=8000,
        learning_rate=0.01,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=RANDOM_STATE + 1000 + fold,
        n_jobs=-1,
        verbosity=-1,
    )


def build_catboost(fold):
    return CatBoostClassifier(
        iterations=1200,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE + fold,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def build_xgboost(fold):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=200,
        n_estimators=8000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=1.5,
        gamma=0.05,
        tree_method="hist",
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )


def build_histgb(fold):
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_STATE + fold,
    )


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"

    if model_arg == "lightgbm":
        lgbm_result = train_with_cv(build_lightgbm, X_enc, y, X_test_enc, "LightGBM")

    elif model_arg == "lightgbm2":
        lgbm2_result = train_with_cv(build_lightgbm_deep, X_enc, y, X_test_enc, "LightGBMDeep")

    elif model_arg == "xgboost":
        xgb_result = train_with_cv(build_xgboost, X_enc, y, X_test_enc, "XGBoost")

    elif model_arg == "histgb":
        hist_result = train_with_cv(build_histgb, X_enc, y, X_test_enc, "HistGB", use_eval_set=False)

    elif model_arg == "catboost":
        X_cat = X.copy()
        X_test_cat = X_test.copy()
        for col in categorical_columns:
            X_cat[col] = X_cat[col].fillna("MISSING").astype(str)
            X_test_cat[col] = X_test_cat[col].fillna("MISSING").astype(str)
        cat_result = train_with_cv(
            build_catboost, X_cat, y, X_test_cat, "CatBoost", use_cat_features=True
        )
