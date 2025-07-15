import numpy as np
from scipy.stats import skewnorm, norm
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─── Globals ───────────────────────────────────────────────────────────────────
DEFAULT_SEED = 42
np.random.seed(DEFAULT_SEED)

def _clip(arr: np.ndarray, grid: float) -> np.ndarray:
    return np.clip(arr, 0, grid)

def generate_uniform_distribution(
    grid_size: float, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform (x,y) in [0,grid_size]."""
    x = np.random.uniform(0, grid_size, num_points)
    y = np.random.uniform(0, grid_size, num_points)
    return _clip(x, grid_size), _clip(y, grid_size)

def generate_skewed_distribution(
    grid_size: float, num_points: int, skewness: float
) -> Tuple[np.ndarray, np.ndarray]:
    """General skewed distribution controlled by skewness (+ve or –ve)."""
    loc = (grid_size/4 if skewness>0 else 3*grid_size/4)
    x = skewnorm.rvs(skewness, loc=loc, scale=grid_size/2, size=num_points)
    y = skewnorm.rvs(skewness, loc=loc, scale=grid_size/2, size=num_points)
    return _clip(x, grid_size), _clip(y, grid_size)

def generate_centered_distribution(
    grid_size: float, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Normal around center with σ=grid_size/8."""
    x = norm.rvs(loc=grid_size/2, scale=grid_size/8, size=num_points)
    y = norm.rvs(loc=grid_size/2, scale=grid_size/8, size=num_points)
    return _clip(x, grid_size), _clip(y, grid_size)

def generate_corner_distribution(
    grid_size: float, num_points: int, corner: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Biased towards one of four corners."""
    corners = {
        'top-right':    (grid_size, grid_size),
        'top-left':     (0,         grid_size),
        'bottom-right': (grid_size, 0),
        'bottom-left':  (0,         0)
    }
    cx, cy = corners.get(corner, (grid_size, grid_size))
    x = norm.rvs(loc=cx, scale=grid_size/8, size=num_points)
    y = norm.rvs(loc=cy, scale=grid_size/8, size=num_points)
    return _clip(x, grid_size), _clip(y, grid_size)

def generate_edge_distribution(
    grid_size: float, num_points: int, edge: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Biased towards one side of the square."""
    if edge in ('top','bottom'):
        x = np.random.uniform(0, grid_size, num_points)
        y = norm.rvs(loc=grid_size if edge=='top' else 0, scale=grid_size/16, size=num_points)
    else:
        y = np.random.uniform(0, grid_size, num_points)
        x = norm.rvs(loc=grid_size if edge=='right' else 0, scale=grid_size/16, size=num_points)
    return _clip(x, grid_size), _clip(y, grid_size)

if __name__ == "__main__":
    # Quick demo plot
    funcs = [
        (generate_uniform_distribution,  'Uniform', {}),
        (lambda g,n: generate_skewed_distribution(g,n,5), 'Pos Skew', {}),
        (lambda g,n: generate_skewed_distribution(g,n,-5),'Neg Skew',{}),
        (generate_centered_distribution,'Centered',{}),
        (lambda g,n: generate_corner_distribution(g,n,'top-right'),'Corner',{}),
        (lambda g,n: generate_edge_distribution(g,n,'top'),'Edge',{})
    ]
    fig, axes = plt.subplots(2,3, figsize=(12,8))
    for ax, (fn, title, _) in zip(axes.flatten(), funcs):
        x,y = fn(100,100)
        ax.scatter(x,y, s=10)
        ax.set_title(title)
        ax.set_xlim(0,100); ax.set_ylim(0,100)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.show()
