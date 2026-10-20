"""
channels.py  --  6G massive-MIMO channel generators (BS = ULA with N antennas, users = single antenna)

Scenarios
---------
UMi-LOS, UMi-NLOS, UMa-LOS, UMa-NLOS, InH-LOS, InH-NLOS, InF-LOS, InF-NLOS
    Clustered far-field model following the 3GPP TR 38.901 procedure (Sec. 7.5, Table 7.5-6 style
    parameters, Table 7.4.1-1 path loss), reduced to a narrow-band azimuth-only channel:
        cluster delays/powers -> K-factor -> cluster AoDs (from ASD) -> 20 rays/cluster -> steering sum.
    Zenith, delay-domain (wideband) and cross-LSP correlation are NOT modelled.
NearField
    Spherical-wave (Fresnel) LOS + scatterers.  Users sit inside/near the Rayleigh distance 2*D^2/lambda.
SNS
    Spatially NON-stationary XL-MIMO: near-field paths, each visible only over a random contiguous
    sub-array (visibility region, VR), following the XL-MIMO literature (not part of 38.901).

Convention
----------
H has shape (K, N); row k is the channel row-vector g_k so that y_k = g_k @ x + n_k.
The channel is scaled so that the noise variance is 1 and each RF chain transmits with power
P_tx / K   ->   |g_k f|^2  is directly an SNR (linear).
Codebook columns are f_b = conj(a(theta_b)) / sqrt(N)  (unit norm).

NOTE: the LSP numbers below are representative values typed from TR 38.901; please cross-check them
against the spec release you cite in your paper (they are all in the SCEN table, one place to edit).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

C0 = 299_792_458.0

# 38.901 Table 7.5-3: C_phi as function of number of clusters, ray offsets (Table 7.5-3)
_CPHI = {4: 0.779, 5: 0.860, 8: 1.018, 10: 1.090, 11: 1.123, 12: 1.146, 14: 1.190,
         15: 1.211, 16: 1.226, 19: 1.273, 20: 1.289, 25: 1.358}
_RAY = np.array([0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551])
_RAY = np.concatenate([_RAY, -_RAY])          # 20 rays per cluster


# ----------------------------------------------------------------------------------------------
#  Scenario table
# ----------------------------------------------------------------------------------------------
@dataclass
class Scn:
    name: str
    family: str            # umi | uma | inh | inf | nf | sns
    los: bool
    h_bs: float
    h_ut: float
    d_min: float           # 2-D distance range of the user drop [m]
    d_max: float


SCEN = {
    "UMi-LOS":  Scn("UMi-LOS",  "umi", True,  10.0, 1.5, 10, 150),
    "UMi-NLOS": Scn("UMi-NLOS", "umi", False, 10.0, 1.5, 10, 150),
    "UMa-LOS":  Scn("UMa-LOS",  "uma", True,  25.0, 1.5, 35, 400),
    "UMa-NLOS": Scn("UMa-NLOS", "uma", False, 25.0, 1.5, 35, 400),
    "InH-LOS":  Scn("InH-LOS",  "inh", True,  3.0,  1.0, 3,  40),
    "InH-NLOS": Scn("InH-NLOS", "inh", False, 3.0,  1.0, 3,  40),
    "InF-LOS":  Scn("InF-LOS",  "inf", True,  8.0,  1.5, 5,  80),
    "InF-NLOS": Scn("InF-NLOS", "inf", False, 8.0,  1.5, 5,  80),
    "NearField": Scn("NearField", "nf", True, 10.0, 1.5, 5, 60),
    "SNS":       Scn("SNS",       "sns", True, 10.0, 1.5, 8, 80),
}
ALL_SCENARIOS = list(SCEN.keys())


def _lsp(s: Scn, fc: float, vs: float = 3.0):
    """Large-scale-parameter statistics.  Returns dict of (mu, sigma) pairs (log10 units, K in dB)."""
    l1 = np.log10(1 + fc)
    lf = np.log10(max(fc, 6.0))
    f, los = s.family, s.los
    if f in ("umi", "nf", "sns"):
        if los:
            return dict(ds=(-0.24 * l1 - 7.14, 0.38), asd=(-0.05 * l1 + 1.21, 0.41), K=(9, 5),
                        rtau=3.0, ncl=12, c_asd=3.0, sf=4.0)
        return dict(ds=(-0.24 * l1 - 6.83, 0.16 * l1 + 0.28), asd=(-0.23 * l1 + 1.53, 0.11 * l1 + 0.33),
                    K=None, rtau=2.1, ncl=19, c_asd=10.0, sf=7.82)
    if f == "uma":
        if los:
            return dict(ds=(-6.955 - 0.0963 * lf, 0.66), asd=(1.06 + 0.1114 * lf, 0.28), K=(9, 3.5),
                        rtau=2.5, ncl=12, c_asd=5.0, sf=4.0)
        return dict(ds=(-6.28 - 0.204 * lf, 0.39), asd=(1.5 - 0.1144 * lf, 0.28), K=None,
                    rtau=2.3, ncl=20, c_asd=2.0, sf=6.0)
    if f == "inh":
        if los:
            return dict(ds=(-0.01 * l1 - 7.692, 0.18), asd=(1.60, 0.18), K=(7, 4),
                        rtau=3.6, ncl=15, c_asd=5.0, sf=3.0)
        return dict(ds=(-0.28 * l1 - 7.173, 0.10 * l1 + 0.055), asd=(1.62, 0.25), K=None,
                    rtau=3.0, ncl=19, c_asd=5.0, sf=8.03)
    if f == "inf":
        if los:
            return dict(ds=(np.log10(26 * vs + 14) - 9.35, 0.15), asd=(1.56 - 0.187 * l1, 0.20), K=(7, 8),
                        rtau=2.7, ncl=25, c_asd=5.0, sf=4.3)
        return dict(ds=(np.log10(30 * vs + 32) - 9.44, 0.19), asd=(1.57 - 0.17 * l1, 0.20), K=None,
                    rtau=3.0, ncl=25, c_asd=5.0, sf=5.7)
    raise ValueError(f)


def path_loss_db(s: Scn, d2d: np.ndarray, fc: float) -> np.ndarray:
    """TR 38.901 Table 7.4.1-1 (UMi-street canyon, UMa, InH-office, InF-SL)."""
    d3 = np.hypot(d2d, s.h_bs - s.h_ut)
    lf = np.log10(fc)
    f = s.family
    if f in ("umi", "uma", "nf", "sns"):
        uma = f == "uma"
        hb, hu = s.h_bs - 1.0, max(s.h_ut - 1.0, 0.5)      # effective heights (h_E = 1 m)
        dbp = 4 * hb * hu * fc * 1e9 / C0
        a, b = (28.0, 22.0) if uma else (32.4, 21.0)
        c2, dcorr = (9.0, 9.0) if uma else (9.5, 9.5)
        pl1 = a + b * np.log10(d3) + 20 * lf
        pl2 = a + 40 * np.log10(d3) + 20 * lf - dcorr * np.log10(dbp ** 2 + (s.h_bs - s.h_ut) ** 2)
        pl_los = np.where(d2d <= dbp, pl1, pl2)
        if s.los:
            return pl_los
        if uma:
            pl_n = 13.54 + 39.08 * np.log10(d3) + 20 * lf - 0.6 * (s.h_ut - 1.5)
        else:
            pl_n = 35.3 * np.log10(d3) + 22.4 + 21.3 * lf - 0.3 * (s.h_ut - 1.5)
        return np.maximum(pl_los, pl_n)
    if f == "inh":
        pl_los = 32.4 + 17.3 * np.log10(d3) + 20 * lf
        if s.los:
            return pl_los
        return np.maximum(pl_los, 38.3 * np.log10(d3) + 17.30 + 24.9 * lf)
    if f == "inf":
        pl_los = 31.84 + 21.5 * np.log10(d3) + 19 * lf
        if s.los:
            return pl_los
        return np.maximum(pl_los, 33.0 + 25.5 * np.log10(d3) + 20 * lf)
    raise ValueError(f)


# ----------------------------------------------------------------------------------------------
#  Array geometry helpers
# ----------------------------------------------------------------------------------------------
def array_positions(N: int, lam: float) -> np.ndarray:
    """ULA along y, half-wavelength spacing, centred at the origin  [m]."""
    return (np.arange(N) - (N - 1) / 2) * lam / 2


def steer_ff(N: int, sin_phi: np.ndarray) -> np.ndarray:
    """Far-field (planar) steering matrix (N, P);  sin_phi: (P,)"""
    y = (np.arange(N) - (N - 1) / 2) * 0.5          # in wavelengths
    return np.exp(1j * 2 * np.pi * np.outer(y, sin_phi))


def steer_nf(N: int, lam: float, r: np.ndarray, theta: np.ndarray, amplitude: bool = True) -> np.ndarray:
    """Spherical-wave steering matrix (N, P) for sources at (r, theta) (theta from broadside)."""
    y = array_positions(N, lam)[:, None]
    r = np.atleast_1d(r)[None, :]
    th = np.atleast_1d(theta)[None, :]
    rn = np.sqrt(r ** 2 + y ** 2 - 2 * r * y * np.sin(th))
    a = np.exp(-1j * 2 * np.pi * (rn - r) / lam)
    return a * (r / rn) if amplitude else a


# ----------------------------------------------------------------------------------------------
#  Codebooks
# ----------------------------------------------------------------------------------------------
def dft_codebook(N: int, oversample: int = 2) -> np.ndarray:
    B = N * oversample
    s = -1 + (2 * np.arange(B) + 1) / B
    return steer_ff(N, s).conj() / np.sqrt(N)                       # (N, B)


def polar_codebook(N: int, fc_ghz: float, oversample: int = 2,
                   dists=(4.0, 8.0, 16.0, 32.0, 64.0)) -> np.ndarray:
    """Polar-domain (angle x distance) near-field codebook + far-field DFT atoms."""
    lam = C0 / (fc_ghz * 1e9)
    B = N * oversample
    s = -1 + (2 * np.arange(B) + 1) / B
    th = np.arcsin(s)
    cols = [steer_ff(N, s).conj()]
    for r in dists:
        cols.append(steer_nf(N, lam, np.full(B, r), th, amplitude=False).conj())
    return np.hstack(cols) / np.sqrt(N)


def make_codebook(scn: str, N: int, fc_ghz: float, oversample: int = 2, kind: str = "auto") -> np.ndarray:
    if kind == "auto":
        kind = "polar" if SCEN[scn].family in ("nf", "sns") else "dft"
    return polar_codebook(N, fc_ghz, oversample) if kind == "polar" else dft_codebook(N, oversample)


# ----------------------------------------------------------------------------------------------
#  Far-field 38.901-style cluster channel for one user
# ----------------------------------------------------------------------------------------------
def _draw(rng, mu_sig):
    return rng.normal(mu_sig[0], mu_sig[1])


def _ff_user(N, lsp, los, sin_los, rng):
    phi_los = np.arcsin(np.clip(sin_los, -1, 1))
    ds = 10 ** _draw(rng, lsp["ds"])
    asd = min(10 ** _draw(rng, lsp["asd"]), 104.0)
    ncl, rt = lsp["ncl"], lsp["rtau"]

    # cluster delays -> powers
    tau = np.sort(-rt * ds * np.log(rng.uniform(1e-9, 1, ncl)))
    tau -= tau.min()
    Z = rng.normal(0, 3.0, ncl)
    P = np.exp(-tau * (rt - 1) / (rt * ds)) * 10 ** (-Z / 10)
    P /= P.sum()
    keep = P > P.max() * 10 ** (-2.5)                       # drop clusters weaker than -25 dB
    P = P[keep] / P[keep].sum()
    n = len(P)

    Kf = 0.0
    K_dB = 0.0
    if los:
        K_dB = _draw(rng, lsp["K"])
        Kf = 10 ** (K_dB / 10)

    # cluster AoDs
    cphi = _CPHI[min(_CPHI, key=lambda k: abs(k - ncl))]
    if los:
        cphi *= 1.1035 - 0.028 * K_dB - 0.002 * K_dB ** 2 + 0.0001 * K_dB ** 3
    phi_p = 2 * (np.deg2rad(asd) / 1.4) * np.sqrt(-np.log(P / P.max())) / cphi
    Xs = rng.choice([-1, 1], n)
    Ys = rng.normal(0, np.deg2rad(asd) / 7, n)
    phi = Xs * phi_p + Ys
    phi = phi - phi[0] + phi_los if los else phi + phi_los   # LOS: strongest cluster aligned with LOS

    # rays
    ang = phi[:, None] + np.deg2rad(lsp["c_asd"]) * _RAY[None, :]       # (n, 20)
    amp = np.sqrt(P[:, None] / len(_RAY)) * np.exp(1j * rng.uniform(0, 2 * np.pi, ang.shape))
    A = steer_ff(N, np.sin(ang).ravel())
    g = A @ (amp.ravel() / np.sqrt(Kf + 1))
    if los:
        g = g + np.sqrt(Kf / (Kf + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * steer_ff(N, np.array([sin_los]))[:, 0]
    return g                                                            # E||g||^2 = N


# ----------------------------------------------------------------------------------------------
#  Near-field and SNS user channel
# ----------------------------------------------------------------------------------------------
def _vr_mask(N, frac, rng):
    L = max(2, int(round(frac * N)))
    c = rng.uniform(L / 2, N - L / 2) if L < N else N / 2
    n = np.arange(N)
    return (np.abs(n - c) <= L / 2).astype(float)


def _nf_user(N, lam, r, sin_th, rng, sns=False, n_sc=6):
    th = np.arcsin(np.clip(sin_th, -1, 1))
    K_dB = rng.normal(9.0, 4.0)
    Kf = 10 ** (K_dB / 10)
    a_los = steer_nf(N, lam, np.array([r]), np.array([th]))[:, 0]
    if sns:
        a_los = a_los * _vr_mask(N, rng.uniform(0.5, 1.0), rng)
    g = np.sqrt(Kf / (Kf + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * a_los
    rs = r * rng.uniform(0.6, 1.8, n_sc)
    ths = th + np.deg2rad(rng.normal(0, 20.0, n_sc))
    ths = np.clip(ths, -np.pi / 2 * 0.98, np.pi / 2 * 0.98)
    A = steer_nf(N, lam, rs, ths)
    if sns:
        A = A * np.stack([_vr_mask(N, rng.uniform(0.15, 0.5), rng) for _ in range(n_sc)], axis=1)
    w = np.exp(1j * rng.uniform(0, 2 * np.pi, n_sc)) * np.sqrt(rng.dirichlet(np.ones(n_sc)) / (Kf + 1))
    return g + A @ w


# ----------------------------------------------------------------------------------------------
#  Public API
# ----------------------------------------------------------------------------------------------
def generate_channels(scn: str, N: int, K: int, rng: np.random.Generator, fc_ghz: float = 28.0,
                      sin_range: float = 0.866, tx_dbm: float = 30.0, bw_hz: float = 100e6,
                      nf_db: float = 9.0, shadowing: bool = True):
    """
    Returns
    -------
    H    : (K, N) complex, noise-normalised (noise variance 1, per-chain power P_tx/K)
    info : dict(sin_los, d2d, pl_db)

    sin_range : half-width of the user drop sector in sin(phi) space, sin(phi_LOS) ~ U(-r, +r).
                0.866 = +-60 deg (normal 120-deg sector); small values = crowded 'stress' drop.
    """
    s = SCEN[scn]
    lam = C0 / (fc_ghz * 1e9)
    sin_los = rng.uniform(-sin_range, sin_range, K)
    d2d = rng.uniform(s.d_min, s.d_max, K)
    pl = path_loss_db(s, d2d, fc_ghz)
    lsp = _lsp(s, fc_ghz)
    if shadowing:
        pl = pl + rng.normal(0, lsp["sf"], K)

    H = np.zeros((K, N), complex)
    for k in range(K):
        if s.family in ("nf", "sns"):
            H[k] = _nf_user(N, lam, d2d[k], sin_los[k], rng, sns=(s.family == "sns"))
        else:
            H[k] = _ff_user(N, lsp, s.los, sin_los[k], rng)

    n0_dbm = -174 + 10 * np.log10(bw_hz) + nf_db
    p_chain_dbm = tx_dbm - 10 * np.log10(K)
    snr_scale_db = p_chain_dbm - pl - n0_dbm                    # per user
    H *= (10 ** (snr_scale_db / 20))[:, None]
    return H, dict(sin_los=sin_los, d2d=d2d, pl_db=pl)


def rayleigh_distance(N: int, fc_ghz: float) -> float:
    lam = C0 / (fc_ghz * 1e9)
    D = (N - 1) * lam / 2
    return 2 * D ** 2 / lam
