import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

RANDOM_STATE = 42
train_df = pd.read_csv("datasets/train.csv")
y = train_df["addicted_label"].astype(int)


def create_features(df):
    df = df.copy()
    eps = 1e-6
    df["known_usage_hours"] = df["social_media_hours"] + df["gaming_hours"] + df["work_study_hours"]
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
    for col in df.columns:
        if col not in ("id", "addicted_label"):
            df[col + "_missing"] = df[col].isna().astype(int)
    df["notifications_per_known_usage"] = df["notifications_per_day"] / (df["known_usage_hours"] + eps)
    df["app_opens_per_known_usage"] = df["app_opens_per_day"] / (df["known_usage_hours"] + eps)
    df["avg_session_length"] = df["daily_screen_time_hours"] / (df["app_opens_per_day"] + eps)
    df["total_screen_week"] = df["daily_screen_time_hours"] * 5 + df["weekend_screen_time"] * 2
    return df


X = create_features(train_df).drop(columns=["addicted_label", "id"])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])
Xp = pre.fit_transform(X)

# single fold split (same as full run fold 1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
tr, va = list(cv.split(Xp, y))[0]
X_tr, X_va, y_tr, y_va = Xp[tr], Xp[va], y.iloc[tr], y.iloc[va]

configs = {
    "base_lr002_lv63": dict(learning_rate=0.02, num_leaves=63, min_child_samples=50, colsample_bytree=0.7),
    "deeper_lr002_lv127": dict(learning_rate=0.02, num_leaves=127, min_child_samples=50, colsample_bytree=0.7),
    "deeper_lr002_lv255": dict(learning_rate=0.02, num_leaves=255, min_child_samples=100, colsample_bytree=0.6),
    "fast_lr004_lv63": dict(learning_rate=0.04, num_leaves=63, min_child_samples=50, colsample_bytree=0.7),
    "slow_lr01_lv255": dict(learning_rate=0.01, num_leaves=255, min_child_samples=80, colsample_bytree=0.5),
}

for name, params in configs.items():
    model = LGBMClassifier(
        objective="binary",
        n_estimators=8000,
        max_depth=-1,
        subsample=0.85,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[early_stopping(200, verbose=False)])
    best_iter = model.best_iteration_
    va_pred = model.predict_proba(X_va)[:, 1]
    tr_pred = model.predict_proba(X_tr)[:, 1]
    print(f"{name}: fold1 AUC={roc_auc_score(y_va, va_pred):.6f} "
          f"train AUC={roc_auc_score(y_tr, tr_pred):.6f} best_iter={best_iter}")
