"""
utils_n.py

Physical channel models and UAV power utilities.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Union, Tuple

# ─── Physical Constants & Params ───────────────────────────────────────────────
C          = 3e8     # Speed of light (m/s)
F_C        = 2e9     # Carrier frequency (Hz)
KSI_LOS    = 1       # LOS additional loss (dB)
KSI_NLOS   = 20      # NLOS additional loss (dB)
C0         = 0.37    # LOS probability coeff
D0         = 0.21    # LOS probability exp
NOISE_DBM  = -120    # Noise level (dBm)
SINR_TH    = 10      # SINR threshold (dB)
MAX_PL     = 100     # Max path‑loss tolerated (dB)
P_HOVER    = 79.99+88.89  # Hover power (W)

# UAV sampling
MIN_ALT    = 50
MAX_ALT    = 5000
UAV_SAMPLES= 50

def probability_of_los(
    r: Union[float, np.ndarray],
    h: Union[float, np.ndarray]
) -> np.ndarray:
    """Vectorized LOS probability (%) vs elevation angle minus 15°."""
    r_arr = np.asarray(r)
    h_arr = np.asarray(h)
    θ = np.arctan2(h_arr, r_arr, where=(r_arr!=0), out=np.full_like(r_arr, np.pi/2))
    α = np.degrees(θ) - 15
    α = np.clip(α, 0, None)
    return C0 * α**D0

def free_space_path_loss(d: Union[float, np.ndarray]) -> np.ndarray:
    """FSPL (dB) for distance d (m)."""
    return 20*np.log10(4*np.pi*F_C*d/C)

def average_path_loss(
    r: Union[float, np.ndarray],
    h: Union[float, np.ndarray]
) -> np.ndarray:
    """Combined LOS/NLOS path loss (dB)."""
    d = np.hypot(r, h)
    p_los = probability_of_los(r, h)
    return free_space_path_loss(d) + p_los*KSI_LOS + (1-p_los)*KSI_NLOS

def min_transmit_power(
    r: float, h: float
) -> float:
    """Min Tx power (dBm) to meet SINR_TH at (r,h)."""
    pl = average_path_loss(r, h)
    return pl + SINR_TH + NOISE_DBM

def spectral_efficiency(
    r: float, h: float, tx_dbm: float=None
) -> float:
    """
    Bits/s/Hz for given tx power at (r,h).
    If tx_dbm None, uses min_transmit_power().
    """
    if tx_dbm is None:
        tx_dbm = min_transmit_power(r, h)
    snr = 10**((tx_dbm - average_path_loss(r,h) - NOISE_DBM)/10)
    return np.log2(1 + snr)

def bandwidth_required(data_rate: float, r: float, h: float) -> float:
    """Hz needed to support data_rate (bps) at (r,h)."""
    return data_rate / spectral_efficiency(r, h)

def find_radius_for_pathloss(
    h: float, max_pl: float=MAX_PL, bounds: Tuple[float,float]=(0,5000)
) -> float:
    """
    Solve for r such that average_path_loss(r,h) ≈ max_pl via 1D minimization.
    """
    f = lambda r: abs(average_path_loss(r, h) - max_pl)
    res = minimize_scalar(f, bounds=bounds, method='bounded')
    return float(res.x)

def find_max_coverage(alts: np.ndarray) -> Tuple[float, float]:
    """
    Over altitudes, return (max_radius, best_altitude).
    """
    radii = [find_radius_for_pathloss(h) for h in alts]
    idx = int(np.argmax(radii))
    return radii[idx], float(alts[idx])

def dbm_to_watt(dbm: float) -> float:
    """Convert dBm → W."""
    return 10**((dbm - 30)/10)

def power_hover() -> float:
    """Hover power (W)."""
    return P_HOVER

def power_horizontal(
    v: float, u: float, p0: float, p1: float,
    vo: float, rho: float, s: float, A: float
) -> float:
    """
    Power model for horizontal flight and comms.
    """
    t1 = p0*(1/v + 3*v/u**2)
    t2 = p1*(np.sqrt(v**(-4) + 0.25*vo**(-4)) - 0.5*vo**(-2))**0.5
    t3 = 0.5*s*rho*A*v**2
    return t1 + t2 + t3
