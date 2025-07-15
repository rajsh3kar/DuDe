import numpy as np
from typing import List, Tuple, Optional, Dict
from utils_n import path_loss

# ─── Config & RNG ──────────────────────────────────────────────────────────────
MIN_ALTITUDE      = 50
MAX_ALTITUDE      = 450
PL_THRESHOLD      = 100
UAV_SAMPLE_POINTS = 50
RNG_SEED          = 42
_rng = np.random.default_rng(RNG_SEED)

DATARATE_RANGES     = [(1e3, 3e3), (3e3, 5e5), (1e6, 3e6)]
DATARATE_PROPORTION = [0.5, 0.3, 0.2]

def distribute_datarates(num_users: int) -> np.ndarray:
    """
    Partition num_users into classes in DATARATE_RANGES
    according to DATARATE_PROPORTION, then sample uniformly.
    """
    counts = np.floor(np.array(DATARATE_PROPORTION)*num_users).astype(int)
    counts[-1] += num_users - counts.sum()
    samples = [_rng.uniform(low, high, size=cnt)
               for (low, high), cnt in zip(DATARATE_RANGES, counts)]
    return np.concatenate(samples)

def find_edge_users(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: float,
    edge_threshold: float
) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    """
    Split (x,y) into edge vs interior based on edge_threshold.
    """
    edge, interior = [], []
    for xi, yi in zip(x, y):
        if xi<=edge_threshold or xi>=grid_size-edge_threshold \
        or yi<=edge_threshold or yi>=grid_size-edge_threshold:
            edge.append((xi, yi))
        else:
            interior.append((xi, yi))
    return edge, interior

def generate_users(
    grid_size: float, user_density: float
) -> Dict[str, np.ndarray]:
    """
    Uniformly scatter users over grid_size² area with datarates.
    """
    n = int(user_density * grid_size**2)
    x = _rng.uniform(0, grid_size, size=n)
    y = _rng.uniform(0, grid_size, size=n)
    return {'x': x, 'y': y, 'datarate': distribute_datarates(n)}

def calculate_optimal_3D(
    edge_positions: List[Tuple[float,float]],
    grid_size: float,
    pl_threshold: float = PL_THRESHOLD
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    For each altitude in [MIN_ALTITUDE,MAX_ALTITUDE) and
    random UAV 2D candidates, return the first (h, [x,y])
    with path_loss ≤ pl_threshold for all edge users.
    """
    uav2d = _rng.uniform(0, grid_size, (UAV_SAMPLE_POINTS, 2))
    for h in range(MIN_ALTITUDE, MAX_ALTITUDE):
        for ux, uy in uav2d:
            distances = [np.hypot(ux-ex, uy-ey) for ex, ey in edge_positions]
            pls = path_loss(np.array(distances), h)
            if np.max(pls) <= pl_threshold:
                return h, np.array([ux, uy])
    return None, None
