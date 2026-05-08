"""Data loading, cleaning, RUL target construction, and windowing."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import COLS, ALL_SENSORS, R_EARLY, WINDOW_SIZE, STRIDE, SEED


def load_subset(name: str, data_dir: str = "data/raw"):
    """Load one C-MAPSS subset (e.g., 'FD001')."""
    train = pd.read_csv(f"{data_dir}/train_{name}.txt", sep=r"\s+",
                        header=None, names=COLS)
    test  = pd.read_csv(f"{data_dir}/test_{name}.txt", sep=r"\s+",
                        header=None, names=COLS)
    rul   = pd.read_csv(f"{data_dir}/RUL_{name}.txt", sep=r"\s+",
                        header=None, names=["RUL"])
    return train, test, rul


def find_constant_sensors(df, threshold: float = 1e-4):
    """Return column names with std below the threshold."""
    stds = df[ALL_SENSORS].std()
    return stds[stds < threshold].index.tolist()


def add_rul_train(df):
    """Add RUL and clipped RUL to a training dataframe (with full trajectories)."""
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL_clipped"] = df["RUL"].clip(upper=R_EARLY)
    return df


def add_rul_test(test_df, rul_df):
    """Add RUL to a test dataframe using the end-of-test RUL values."""
    max_cycle = test_df.groupby("unit_id")["cycle"].transform("max")
    # rul_df is 0-indexed: engine 1 -> rul_df.loc[0]
    rul_map = {uid: rul_df.loc[uid - 1, "RUL"] for uid in test_df["unit_id"].unique()}
    end_rul = test_df["unit_id"].map(rul_map)
    test_df["RUL"] = end_rul + (max_cycle - test_df["cycle"])
    test_df["RUL_clipped"] = test_df["RUL"].clip(upper=R_EARLY)
    return test_df


def minmax_scale(train_df, test_df, sensor_cols):
    """Fit MinMaxScaler on train sensors, apply to both.

    Warning: this is the SIMPLE version. Use it for FD001 and FD003 which have
    a single operating condition. For FD002 and FD004, use condition_normalize()
    instead — otherwise you are scaling sensors across regimes that have very
    different baseline levels, and the features will mostly capture which
    regime was flown rather than engine degradation.
    """
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols]  = scaler.transform(test_df[sensor_cols])
    return train_df, test_df, scaler


def condition_normalize(train_df, test_df, sensor_cols, n_conditions: int = 6,
                        seed: int = SEED):
    """
    Per-operating-condition min-max scaling for FD002 and FD004.

    Steps:
      1. Fit K-means on the three setting columns from the training data to
         discover the n_conditions distinct flight regimes.
      2. Assign every row in train and test to its nearest cluster.
      3. Within each cluster, fit a separate MinMaxScaler on the training
         sensors and apply it to both train and test rows of that cluster.

    After this step, `sensor_2 = 0.7` means the same thing (relative to that
    regime's healthy baseline) regardless of altitude or throttle.

    Returns
    -------
    train_df, test_df : scaled copies
    cluster_model     : fitted KMeans (to re-use on fresh data)
    scalers           : dict condition_id -> MinMaxScaler
    """
    from sklearn.cluster import KMeans

    settings = [f"setting_{i}" for i in (1, 2, 3)]
    train_df = train_df.copy()
    test_df  = test_df.copy()

    from sklearn.preprocessing import StandardScaler
    setting_scaler = StandardScaler()
    train_settings = setting_scaler.fit_transform(train_df[settings])
    test_settings = setting_scaler.transform(test_df[settings])

    print("Settings before scaling:")
    print(train_df[settings].describe().loc[["mean", "std"]])
    print("Settings after scaling:")
    print(pd.DataFrame(train_settings, columns=settings).describe().loc[["mean", "std"]])

    km = KMeans(n_clusters=n_conditions, random_state=seed, n_init=10)
    train_df["condition"] = km.fit_predict(train_settings)
    test_df["condition"] = km.predict(test_settings)

    scalers = {}
    for c in range(n_conditions):
        sc = MinMaxScaler()
        tr_mask = train_df["condition"] == c
        te_mask = test_df["condition"] == c
        if tr_mask.any():
            train_df.loc[tr_mask, sensor_cols] = sc.fit_transform(
                train_df.loc[tr_mask, sensor_cols])
            scalers[c] = sc
        if te_mask.any() and c in scalers:
            test_df.loc[te_mask, sensor_cols] = scalers[c].transform(
                test_df.loc[te_mask, sensor_cols])

    return train_df, test_df, km, scalers


def split_engines(train_df, val_frac: float = 0.2, seed: int = SEED):
    """Split the training dataframe into train/val by engine ID."""
    unit_ids = train_df["unit_id"].unique()
    train_ids, val_ids = train_test_split(unit_ids, test_size=val_frac,
                                          random_state=seed)
    train_part = train_df[train_df["unit_id"].isin(train_ids)].copy()
    val_part   = train_df[train_df["unit_id"].isin(val_ids)].copy()
    return train_part, val_part


def make_windows(df, sensor_cols, window: int = WINDOW_SIZE, stride: int = STRIDE,
                 include_settings: bool = False, include_condition: bool = False):
    """
    Build fixed-length windows per engine.

    Parameters
    ----------
    df : DataFrame containing at least [unit_id, cycle, RUL_clipped, sensor_cols...]
    include_settings : if True, also return a (n_windows, window, 3) array of
                       the raw setting values during each window.
    include_condition : if True, include a 'condition' column in meta (the
                        condition label from condition_normalize).

    Returns
    -------
    X : ndarray, shape (n_windows, window, n_sensors)
    y : ndarray, shape (n_windows,)
    meta : ndarray, shape (n_windows, k) -- columns [unit_id, cycle] plus
           'condition' if include_condition=True
    settings : ndarray (n_windows, window, 3) if include_settings else None
    """
    has_condition = include_condition and "condition" in df.columns
    Xs, ys, meta, Ss = [], [], [], []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        arr = g[sensor_cols].to_numpy()
        rul = g["RUL_clipped"].to_numpy()
        cyc = g["cycle"].to_numpy()
        settings_arr = g[[f"setting_{i}" for i in (1, 2, 3)]].to_numpy() \
            if include_settings else None
        cond = g["condition"].to_numpy() if has_condition else None
        if len(arr) < window:
            pad_n = window - len(arr)
            arr = np.vstack([np.tile(arr[0], (pad_n, 1)), arr])
            rul = np.concatenate([np.full(pad_n, rul[0]), rul])
            cyc = np.concatenate([np.full(pad_n, cyc[0]), cyc])
            if settings_arr is not None:
                settings_arr = np.vstack([np.tile(settings_arr[0], (pad_n, 1)),
                                          settings_arr])
            if cond is not None:
                cond = np.concatenate([np.full(pad_n, cond[0]), cond])
        for end in range(window, len(arr) + 1, stride):
            Xs.append(arr[end - window:end])
            ys.append(rul[end - 1])
            if has_condition:
                # Use the dominant condition in the window as the label
                window_conds = cond[end - window:end]
                dominant = int(np.bincount(window_conds).argmax())
                meta.append([uid, cyc[end - 1], dominant])
            else:
                meta.append([uid, cyc[end - 1]])
            if settings_arr is not None:
                Ss.append(settings_arr[end - window:end])

    X = np.array(Xs)
    y = np.array(ys)
    meta = np.array(meta)
    S = np.array(Ss) if include_settings else None
    return X, y, meta, S


def make_last_window_per_engine(df, sensor_cols, window: int = WINDOW_SIZE,
                                include_settings: bool = False,
                                include_condition: bool = False):
    """For each engine, return the single last window (used for test evaluation)."""
    has_condition = include_condition and "condition" in df.columns
    Xs, ys, ids, conds, Ss = [], [], [], [], []
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        arr = g[sensor_cols].to_numpy()
        rul = g["RUL_clipped"].to_numpy()
        if len(arr) < window:
            pad_n = window - len(arr)
            arr = np.vstack([np.tile(arr[0], (pad_n, 1)), arr])
            rul = np.concatenate([np.full(pad_n, rul[0]), rul])
        Xs.append(arr[-window:])
        ys.append(rul[-1])
        ids.append(uid)
        if has_condition:
            window_conds = g["condition"].to_numpy()[-window:]
            conds.append(int(np.bincount(window_conds).argmax()))
        if include_settings:
            settings_arr = g[[f"setting_{i}" for i in (1, 2, 3)]].to_numpy()
            if len(settings_arr) < window:
                pad_n = window - len(settings_arr)
                settings_arr = np.vstack([np.tile(settings_arr[0], (pad_n, 1)),
                                          settings_arr])
            Ss.append(settings_arr[-window:])

    X = np.array(Xs)
    y = np.array(ys)
    ids = np.array(ids)
    S = np.array(Ss) if include_settings else None
    if has_condition:
        return X, y, ids, np.array(conds), S
    return X, y, ids, None, S


def extract_baseline_window(train_df, sensor_cols, n_cycles: int = 30):
    """
    For every engine in train_df, return the FIRST n_cycles as a 'healthy baseline'.
    Returns a dict: unit_id -> ndarray of shape (n_cycles, n_sensors).
    """
    baselines = {}
    for uid, g in train_df.groupby("unit_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        if len(g) < n_cycles:
            continue
        baselines[uid] = g[sensor_cols].iloc[:n_cycles].to_numpy()
    return baselines
