"""Dataset adapter using the supplied channels.py propagation model.

Rows specify the user-specific scenario, carrier frequency, array size, 2D
user distance, LOS azimuth sine, and antenna heights. The original channels.py
provides path loss, scenario LSP statistics, stochastic scattering and codebooks.
The user's row-specific distance and angle are preserved (not re-drawn).

``external`` demands channels.py; ``auto`` uses it when installed and otherwise
warns and uses the explicitly named ``simplified`` fallback.
"""
import importlib
import warnings
import numpy as np
import pandas as pd

CHANNEL_MODULE = "channels"
SHEET = "Channel_Generator_Input"
CHANNEL_MODEL = "auto"    # "auto" / "external" / "simplified"
FALLBACK_WARNING_EMITTED = False
# The supplied channels.py computes path loss and shadow sigma itself.


def load_dataset(path, sheet=SHEET):
    df = pd.read_excel(path, sheet_name=sheet)
    need = {"set", "N", "scenario", "fc_GHz", "d2d_m", "sin_phi_LOS"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Dataset sheet '{sheet}' is missing columns: {sorted(missing)}")
    if df.empty or df["scenario"].isna().any():
        raise ValueError("Dataset contains no rows or missing scenario names")
    return df


def choose_sets(df, selection):
    names = list(dict.fromkeys(df["set"]))
    if selection == "all":
        chosen = names
    elif isinstance(selection, str):
        chosen = [selection]
    else:
        chosen = list(selection)
    if not set(chosen).issubset(set(names)):
        raise ValueError(f"Unknown set in {chosen}; available: {names}")
    return {name: df[df["set"] == name].reset_index(drop=True) for name in chosen}


def _external_available(rows):
    try:
        h = importlib.import_module(CHANNEL_MODULE)
    except ImportError:
        return False
    return all(hasattr(h, key) for key in (
        "SCEN", "_lsp", "_ff_user", "_nf_user", "path_loss_db", "C0",
        "dft_codebook", "polar_codebook"))


def _helpers():
    try:
        return importlib.import_module(CHANNEL_MODULE)
    except ImportError as exc:
        raise ImportError(f"Missing '{CHANNEL_MODULE}.py'; for original channels supply "
                          "the file or use --channel-model simplified") from exc


def _dft_codebook(N, oversample):
    # Steering vectors indexed on a uniform grid in spatial frequency [-1,1).
    s = np.linspace(-1, 1, N * oversample, endpoint=False)
    return np.exp(1j * np.pi * np.outer(np.arange(N), s)) / np.sqrt(N)


def _fallback_codebook(kind, N, fc_ghz, oversample):
    F = _dft_codebook(N, oversample)
    if kind == "dft":
        return F
    if kind != "polar":
        raise ValueError("codebook must be 'polar' or 'dft'")
    # Add range-dependent quadratic Fresnel phase atoms for near-field channels.
    # The near-field onset is set relative to array length at carrier wavelength.
    lam = 299792458.0 / (fc_ghz * 1e9)
    spacing = lam / 2
    pos = (np.arange(N) - (N - 1) / 2) * spacing
    ranges = (max(5.0, 0.25 * (N * spacing)**2 / lam),
              max(10.0, 2 * (N * spacing)**2 / lam))
    angles = np.linspace(-1, 1, max(8, N // 2), endpoint=False)
    cols = []
    for r in ranges:
        for s in angles:
            phase = 2 * np.pi / lam * (pos * s - pos**2 * (1 - s*s) / (2 * r))
            cols.append(np.exp(1j*phase)/np.sqrt(N))
    return np.column_stack([F, np.asarray(cols, complex).T])


def get_codebook(kind, N, fc_ghz, oversample, mode=None):
    mode = CHANNEL_MODEL if mode is None else mode
    if mode not in ("auto", "simplified", "external"):
        raise ValueError("Invalid channel model setting")
    if mode != "simplified":
        try:
            h = _helpers()
            fn = h.polar_codebook if kind == "polar" else h.dft_codebook
            return fn(N, fc_ghz, oversample) if kind == "polar" else fn(N, oversample)
        except (ImportError, AttributeError):
            if mode == "external":
                raise
    return _fallback_codebook(kind, N, fc_ghz, oversample)


def _fallback_channels(rows, N, rng, n_chains, tx_dbm, bw_hz, nf_db, shadowing):
    K = len(rows)
    H = np.zeros((K, N), complex)
    pl = np.zeros(K)
    scenarios = []
    for k, (_, r) in enumerate(rows.iterrows()):
        sc = str(r["scenario"])
        d = max(float(r["d2d_m"]), 1.0)
        fc = float(r["fc_GHz"])
        sin_phi = np.clip(float(r["sin_phi_LOS"]), -1, 1)
        lam = 299792458.0 / (fc * 1e9)
        z = np.arange(N)
        # Free-space reference loss plus transparent, scenario-dependent add-on.
        # THESE are approximate simulation assumptions, NOT workbook measurements.
        extra_loss = (0 if "LOS" in sc and "NLOS" not in sc else 14)
        if sc.startswith("InF"):
            extra_loss += 6
        if sc == "SNS":
            extra_loss += 8
        pl[k] = 32.4 + 20*np.log10(fc) + 20*np.log10(d) + extra_loss
        if shadowing:
            pl[k] += rng.normal(0.0, 3.0 if extra_loss == 0 else 5.0)
        # Each channel has directional component and independent scattered rays.
        a = np.exp(-1j*np.pi*z*sin_phi) / np.sqrt(N)
        if sc in ("NearField", "SNS"):
            pos = (z - (N-1)/2) * lam/2
            phase = -2j*np.pi/lam * pos**2 * (1-sin_phi**2) / (2*d)
            a = a * np.exp(phase)
        rays = np.zeros(N, complex)
        for _ in range(6):
            s = np.clip(sin_phi + rng.normal(0, 0.2), -1, 1)
            rays += (rng.standard_normal() + 1j*rng.standard_normal()) / np.sqrt(12*6) * (
                np.exp(-1j*np.pi*z*s) / np.sqrt(N))
        los = "LOS" in sc and "NLOS" not in sc
        H[k] = (np.sqrt(0.85)*a + np.sqrt(0.15)*rays if los else rays)
        if sc == "SNS":
            center = rng.uniform(0, N)
            H[k] *= np.exp(-0.5*((z-center)/(max(N/7, 1)))**2)
        scenarios.append(sc)
    noise_dbm = -174.0 + 10*np.log10(bw_hz) + nf_db
    p_chain_dbm = tx_dbm - 10*np.log10(n_chains)
    snr_scale = 10**((p_chain_dbm - pl - noise_dbm)/20)
    H *= snr_scale[:, None]
    info = dict(sin_los=rows["sin_phi_LOS"].to_numpy(float),
                d2d=rows["d2d_m"].to_numpy(float), pl_db=pl,
                scenario=scenarios, channel_model="simplified")
    return H, info


def generate_channels_from_rows(rows, N, rng, n_chains, tx_dbm=30, bw_hz=100e6,
                                nf_db=9, shadowing=True, pathloss_source="dataset",
                                model=None):
    global FALLBACK_WARNING_EMITTED
    mode = CHANNEL_MODEL if model is None else model
    if mode not in ("auto", "external", "simplified"):
        raise ValueError("model must be auto, external or simplified")
    use_external = mode == "external" or (mode == "auto" and _external_available(rows))
    if use_external:
        if not _external_available(rows):
            raise RuntimeError("--channel-model external requires the original channels.py "
                               "in the same folder as dataset_channels.py")
        from dataclasses import replace
        h = _helpers()
        H = np.zeros((len(rows), N), dtype=complex)
        pl = np.zeros(len(rows), dtype=float)
        scenarios = []
        for k, (_, r) in enumerate(rows.iterrows()):
            sc = str(r["scenario"])
            if sc not in h.SCEN:
                raise ValueError(f"Unknown scenario in channels.py: {sc}")
            # Preserve user-supplied antenna heights rather than silently using
            # defaults in SCEN when the workbook overrides them.
            cfg = replace(h.SCEN[sc], h_bs=float(r.get("h_bs_m", h.SCEN[sc].h_bs)),
                          h_ut=float(r.get("h_ut_m", h.SCEN[sc].h_ut)))
            fc = float(r["fc_GHz"])
            d = float(r["d2d_m"])
            angle = float(r["sin_phi_LOS"])
            if fc <= 0 or d <= 0 or not -1 <= angle <= 1:
                raise ValueError(f"Invalid frequency/distance/sin_phi_LOS at dataset row {r.name}")
            lsp = dict(h._lsp(cfg, fc))
            # Optional measurement-based LSP overrides if a richer workbook is used.
            if pd.notna(r.get("delay_spread_ns", np.nan)) and r["delay_spread_ns"] > 0:
                lsp["ds"] = (np.log10(float(r["delay_spread_ns"]) * 1e-9), 0.0)
            if pd.notna(r.get("ASD_deg", np.nan)) and r["ASD_deg"] > 0:
                lsp["asd"] = (np.log10(float(r["ASD_deg"])), 0.0)
            if pd.notna(r.get("K_factor_dB", np.nan)) and cfg.los:
                lsp["K"] = (float(r["K_factor_dB"]), 0.0)

            if pathloss_source == "dataset" and pd.notna(r.get("path_loss_dB", np.nan)):
                pl[k] = float(r["path_loss_dB"])
            else:
                pl[k] = float(h.path_loss_db(cfg, np.asarray([d]), fc)[0])
            if shadowing:
                # If the sheet has its own shadow sigma, use it. Otherwise
                # use the scenario-dependent standard deviation in channels.py.
                sf = r.get("shadow_fading_sigma_dB", np.nan)
                sigma = float(sf) if pd.notna(sf) else float(lsp["sf"])
                pl[k] += rng.normal(0.0, sigma)

            if cfg.family in ("nf", "sns"):
                H[k] = h._nf_user(N, h.C0 / (fc * 1e9), d, angle, rng,
                                  sns=(cfg.family == "sns"))
            else:
                H[k] = h._ff_user(N, lsp, cfg.los, angle, rng)
            scenarios.append(sc)
        noise_dbm = -174.0 + 10 * np.log10(bw_hz) + nf_db
        p_chain_dbm = tx_dbm - 10 * np.log10(len(rows))
        H *= (10 ** ((p_chain_dbm - pl - noise_dbm) / 20.0))[:, None]
        return H, dict(sin_los=rows["sin_phi_LOS"].to_numpy(float),
                       d2d=rows["d2d_m"].to_numpy(float), pl_db=pl,
                       scenario=scenarios, channel_model="external")
    if not FALLBACK_WARNING_EMITTED:
        warnings.warn("Original channels.py was not available. "
                      "Using simplified geometric channel model; these are illustrative "
                      "and NOT results from the original 3GPP model.", RuntimeWarning)
        FALLBACK_WARNING_EMITTED = True
    return _fallback_channels(rows, N, rng, n_chains, tx_dbm, bw_hz, nf_db, shadowing)
