import _path  # noqa: F401  # adds src/ to sys.path
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

import xgboost as xgb
# Quick test that XGBoost's GPU path works
try:
    dtrain = xgb.DMatrix([[1.0], [2.0]], label=[0.0, 1.0])
    xgb.train({"device": "cuda", "tree_method": "hist"}, dtrain, num_boost_round=1)
    print("XGBoost GPU: works")
except Exception as e:
    print(f"XGBoost GPU failed: {e}")