import numpy as np
import pandas as pd

blend = np.load("blend_result.npz")
test_pred = blend["test"]

test_df = pd.read_csv("datasets/test.csv")
submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "addicted_label": np.clip(test_pred, 0, 1),
    }
)

assert submission.columns.tolist() == ["id", "addicted_label"]
assert len(submission) == len(test_df)
assert submission["id"].equals(test_df["id"])
assert submission["addicted_label"].between(0, 1).all()
assert submission["addicted_label"].nunique() > 2

submission.to_csv("submission.csv", index=False, header=True)
print("submission.csv written:", submission.shape)
print(submission["addicted_label"].describe().to_string())
