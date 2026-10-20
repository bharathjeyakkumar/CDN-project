"""
dataset_channels.py -- channels driven by channel_test_suite_organized.xlsx

What this file does
-------------------
1. load_dataset()          reads the sheet "Channel_Generator_Input" (300 rows, one row = one user).
2. choose_sets()           picks "all" or one of "Set 1" / "Set 2" / "Set 3".
3. draw_snapshots()        builds snapshots: K_DROP users drawn at random from ONE set
                           (scenarios are mixed, as requested). A snapshot has one antenna
                           count N (64 or 128) because N is a property of the set.
4. generate_channels_from_rows()
                           your generate_channels() logic, but distance, angle, path loss and
                           shadow-fading sigma come from the spreadsheet row instead of random draws.

Changes compared with your generate_channels()
----------------------------------------------
* d2d, sin(phi_LOS), scenario, fc come from the row (not drawn at random).
* Path loss: the spreadsheet's path_loss_dB is the plain 3GPP formula WITHOUT shadow fading
  (see sheet "Formulas"), so shadowing is added here with the row's shadow_fading_sigma_dB.
  Set PATHLOSS_SOURCE = "model" to recompute path loss with your own path_loss_db() instead.
* BUG FIX in the last step. Your line
        H = (10 * (snr_scale_db / 20))[:, None]
  discards the channel and uses 10*x instead of 10**x. It is now
        H = H * 10 ** (snr_scale_db / 20)[:, None]
* Per-chain transmit power is P_tx / n_chains, where n_chains = number of RF chains that are
  actually active in the QAOA subproblem (your function used K).

Your helper functions
---------------------
SCEN, C0, path_loss_db, _lsp, _nf_user and _ff_user are NOT defined here. They are imported from the
module named CHANNEL_MODULE (the file that contains your generate_channels). Change the name below
if your file is called something else. The call signatures are exactly the ones used in your function:
    SCEN[scn] -> object with .family ("nf", "sns", other), .los, .d_min, .d_max
    path_loss_db(s, d2d_array, fc_ghz)
    _lsp(s, fc_ghz) -> dict with at least "sf"
    _nf_user(N, lam, d2d, sin_los, rng, sns=bool)
    _ff_user(N, lsp, los, sin_los, rng)
"""

import importlib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
CHANNEL_MODULE = "channels"      # <- the file that holds your generate_channels and helpers
SHEET = "Channel_Generator_Input"

# dataset scenario name -> key used in your SCEN dictionary (identity unless you say otherwise)
SCENARIO_ALIASES = {}

# Optional: overwrite entries of the dict returned by _lsp() with the sampled per-row values of the
# spreadsheet, e.g. {"ds": "delay_spread_ns", "asd": "ASD_deg", "kf": "K_factor_dB"}.
# Keys on the left must be the names your _lsp() uses. Empty = let _lsp() decide (default).
LSP_KEY_MAP = {}

# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------

def load_dataset(path, sheet=SHEET):
    df = pd.read_excel(path, sheet_name=sheet)
    need = {"set", "N", "R", "scenario", "fc_GHz", "d2d_m", "sin_phi_LOS",
            "path_loss_dB", "shadow_fading_sigma_dB"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"sheet '{sheet}' is missing columns: {sorted(missing)}")
    return df


def choose_sets(df, selection):
    """selection: "all", one name such as "Set 2", or a list of names. Returns {set_name: rows}."""
    names = sorted(df["set"].unique())
    if selection == "all":
        chosen = names
    elif isinstance(selection, str):
        if selection not in names:
            raise ValueError(f"unknown set '{selection}', choose from {names} or 'all'")
        chosen = [selection]
    else:
        chosen = list(selection)
    return {n: df[df["set"] == n].reset_index(drop=True) for n in chosen}


def draw_snapshots(sets, snapshots_per_set, k_drop, rng):
    """
    One snapshot = k_drop users drawn at random (without replacement) from one set.
    Returns a list of dicts: set, N, R, rows (DataFrame with k_drop rows).
    """
    snaps = []
    for name, rows in sets.items():
        N, R = int(rows["N"].iloc[0]), int(rows["R"].iloc[0])
        if k_drop > len(rows):
            raise ValueError(f"K_DROP={k_drop} is larger than the {len(rows)} rows of {name}")
        for _ in range(snapshots_per_set):
            pick = rng.choice(len(rows), size=k_drop, replace=False)
            snaps.append(dict(set=name, N=N, R=R, rows=rows.iloc[pick].reset_index(drop=True)))
    return snaps


# ----------------------------------------------------------------------
# channel generation
# ----------------------------------------------------------------------

def _helpers():
    try:
        return importlib.import_module(CHANNEL_MODULE)
    except ImportError as e:
        raise ImportError(
            f"Could not import '{CHANNEL_MODULE}'. Put the file that defines SCEN, C0, path_loss_db, "
            f"_lsp, _nf_user and _ff_user in the same folder and set CHANNEL_MODULE in "
            f"dataset_channels.py to its name (without .py)."
        ) from e


def generate_channels_from_rows(rows, N, rng, n_chains, tx_dbm=30.0, bw_hz=100e6, nf_db=9.0,
                                shadowing=True, pathloss_source="dataset"):
    """
    Noise-normalised channels for the users described by `rows`.

    Returns
    -------
    H    : (K, N) complex, noise variance 1, per-chain power P_tx / n_chains
    info : dict(sin_los, d2d, pl_db, scenario)
    """
    h = _helpers()
    K = len(rows)
    H = np.zeros((K, N), complex)
    pl = np.zeros(K)
    scenarios = []

    for k, r in rows.iterrows():
        scn = SCENARIO_ALIASES.get(r["scenario"], r["scenario"])
        s = h.SCEN[scn]
        fc = float(r["fc_GHz"])
        lam = h.C0 / (fc * 1e9)
        d2d, sin_los = float(r["d2d_m"]), float(r["sin_phi_LOS"])

        lsp = dict(h._lsp(s, fc))
        for key, col in LSP_KEY_MAP.items():
            lsp[key] = float(r[col])

        if pathloss_source == "dataset":
            pl_k = float(r["path_loss_dB"])
        else:
            pl_k = float(np.asarray(h.path_loss_db(s, np.array([d2d]), fc))[0])
        if shadowing:
            pl_k += rng.normal(0.0, float(r["shadow_fading_sigma_dB"]))
        pl[k] = pl_k

        if s.family in ("nf", "sns"):
            H[k] = h._nf_user(N, lam, d2d, sin_los, rng, sns=(s.family == "sns"))
        else:
            H[k] = h._ff_user(N, lsp, s.los, sin_los, rng)
        scenarios.append(scn)

    n0_dbm = -174 + 10 * np.log10(bw_hz) + nf_db
    p_chain_dbm = tx_dbm - 10 * np.log10(n_chains)
    snr_scale_db = p_chain_dbm - pl - n0_dbm                  # per user
    H = H * (10 ** (snr_scale_db / 20))[:, None]              # bug fix: multiply, and 10**x

    info = dict(sin_los=rows["sin_phi_LOS"].to_numpy(float), d2d=rows["d2d_m"].to_numpy(float),
                pl_db=pl, scenario=scenarios)
    return H, info