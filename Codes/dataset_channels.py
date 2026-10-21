"""
dataset_channels.py -- channels driven by channel_test_suite_organized.xlsx

What this file does
-------------------
1. load_dataset()          reads the sheet "Channel_Generator_Input" (300 rows, one row = one user).
2. choose_sets()           picks "all" or one of "Set 1" / "Set 2" / "Set 3".
3. draw_snapshots()        builds snapshots: K_DROP users drawn at random from ONE set
                           (scenarios are mixed). A snapshot has one antenna count N (64 or 128)
                           because N is a property of the set.
4. generate_channels_from_rows()
                           the logic of your generate_channels(), but distance, angle, path loss,
                           shadow-fading sigma and the sampled large-scale parameters come from
                           the spreadsheet row instead of random draws.
5. get_codebook()          your dft_codebook / polar_codebook.

Differences compared with generate_channels() in channels.py
------------------------------------------------------------
* d2d, sin(phi_LOS), scenario and fc come from the row (not drawn at random).
* Path loss: the spreadsheet's path_loss_dB is the plain 3GPP formula WITHOUT shadow fading
  (sheet "Formulas"; it agrees with your path_loss_db() to 0.01 dB), so shadowing is added here with
  the row's shadow_fading_sigma_dB. PATHLOSS_SOURCE = "model" recomputes it with path_loss_db().
* Large-scale parameters: the row's sampled delay_spread_ns, ASD_deg and K_factor_dB replace the
  random draws inside _ff_user (USE_ROW_LSP = True). This is done by passing (value, 0.0) as the
  (mean, sigma) pair, so the same value is "drawn" every time. Near-field / SNS users do not use them.
* Per-chain transmit power is P_tx / n_chains, where n_chains = number of RF chains that are
  actually active in the QAOA subproblem (generate_channels used K).
* The scaling step is the same as in channels.py: H *= 10 ** (snr_scale_db / 20).

Everything else (38.901-style cluster model, near-field and SNS users, codebooks) is your code, imported
from CHANNEL_MODULE, which is "channels" (your channels.py).
"""

import importlib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
CHANNEL_MODULE = "channels"           # your channels.py (without .py)
SHEET = "Channel_Generator_Input"

# dataset scenario name -> key used in your SCEN dictionary (identity: the names already match)
SCENARIO_ALIASES = {}

# True: use the sampled delay spread, angular spread and K-factor of each spreadsheet row
# instead of letting _ff_user draw new random ones.
USE_ROW_LSP = True


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
            raise ValueError(f"{k_drop} users per snapshot is more than the {len(rows)} rows of {name}")
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
            f"Could not import '{CHANNEL_MODULE}'. Put your channels.py in the same folder, or set "
            f"CHANNEL_MODULE in dataset_channels.py to the right file name (without .py)."
        ) from e


def _apply_row_lsp(lsp, r, los):
    """Pin _ff_user's random draws to the row's sampled values: (mean, sigma=0) always returns the mean."""
    ds_ns, asd, kf = r.get("delay_spread_ns"), r.get("ASD_deg"), r.get("K_factor_dB")
    if ds_ns is not None and np.isfinite(ds_ns) and ds_ns > 0:
        lsp["ds"] = (float(np.log10(ds_ns * 1e-9)), 0.0)          # log10 of seconds
    if asd is not None and np.isfinite(asd) and asd > 0:
        lsp["asd"] = (float(np.log10(asd)), 0.0)                   # log10 of degrees
    if los and lsp.get("K") is not None and kf is not None and np.isfinite(kf):
        lsp["K"] = (float(kf), 0.0)                                # dB


def get_codebook(kind, N, fc_ghz, oversample):
    """
    "dft"   : far-field DFT codebook (your dft_codebook), N * oversample beams
    "polar" : far-field DFT atoms PLUS near-field (angle x distance) atoms (your polar_codebook).
              It contains the DFT beams, so far-field users lose nothing; near-field / SNS users gain.
    Columns are unit norm and follow your convention: gain = |g_k f|^2 = |H @ F|^2.
    """
    h = _helpers()
    if kind == "polar":
        return h.polar_codebook(N, fc_ghz, oversample)
    if kind == "dft":
        return h.dft_codebook(N, oversample)
    raise ValueError(f"unknown codebook kind '{kind}'")


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
        if USE_ROW_LSP:
            _apply_row_lsp(lsp, r, s.los)

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
    H = H * (10 ** (snr_scale_db / 20))[:, None]

    info = dict(sin_los=rows["sin_phi_LOS"].to_numpy(float), d2d=rows["d2d_m"].to_numpy(float),
                pl_db=pl, scenario=scenarios)
    return H, info