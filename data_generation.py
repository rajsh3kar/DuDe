import torch
from torch.utils.data import Dataset
import numpy as np
from tester import generate_users, find_edge_users, calculate_optimal_3D

# ─── Config ────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
USER_DENSITY = 0.0005
GRID_SIZE    = 128
NUM_SAMPLES  = 20000
PL_THRESHOLD = 100

class UAVDataset(Dataset):
    """Generates and returns one sample at a time."""
    def __init__(self, n_samples: int, grid_size: int, density: float):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.density   = density

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        while True:
            env = int(np.random.uniform(50, 500))
            users = generate_users(env, self.density)
            edge, _ = find_edge_users(users['x_positions'], users['y_positions'], env, 10)
            alti, hori = calculate_optimal_3D(edge, env, pl_threshold=PL_THRESHOLD)
            if alti is not None:
                break

        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        rates = users['datarates']
        rmin, rmax = rates.min(), rates.max()
        for x, y, rate in zip(users['x_positions'], users['y_positions'], rates):
            i = min(int(x/env*self.grid_size), self.grid_size-1)
            j = min(int(y/env*self.grid_size), self.grid_size-1)
            norm = (rate-rmin)/(rmax-rmin) if rmax>rmin else 0.5
            grid[i, j] += 1 + norm

        target = torch.tensor([hori[0]/env, hori[1]/env, alti/env], dtype=torch.float32)
        env_meta = torch.tensor(env, dtype=torch.float32)
        return torch.from_numpy(grid).unsqueeze(0), target, env_meta

if __name__ == "__main__":
    ds = UAVDataset(NUM_SAMPLES, GRID_SIZE, USER_DENSITY)
    # Option: save the Dataset object for lazy loading
    torch.save(ds, 'uav_dataset.pt')
    print("Saved dataset object to uav_dataset.pt")
