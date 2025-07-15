import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models
from functions_util import generate_users, find_edge_users, calculate_optimal_3D

# ─── Reproducibility & Constants ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

USER_DENSITY     = 0.0005
GRID_SIZE        = 128
BATCH_SIZE       = 128
NUM_SAMPLES      = 10000
TRAIN_RATIO      = 0.8
DROP_RATE        = 0.3
LEARNING_RATE    = 1e-3
EPOCHS           = 20
SCHEDULER_PATIENCE = 5
PL_THRESHOLD     = 100
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UAVDataset(Dataset):
    """Dataset of generated user‐density grids with optimal UAV placement targets."""
    def __init__(self, num_samples: int, grid_size: int, density: float):
        self.num_samples = num_samples
        self.grid_size   = grid_size
        self.density     = density

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # 1) Sample environment size and users
        env = int(np.random.uniform(50, 500))
        users = generate_users(env, self.density)
        edge, _ = find_edge_users(users['x'], users['y'], env, edge_threshold=10)

        # 2) Find a valid (alt, loc); retry with a loop
        while True:
            alti, hori = calculate_optimal_3D(edge, env, pl_threshold=PL_THRESHOLD)
            if alti is not None:
                break

        # 3) Build input grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        rates = users['datarate']
        rmin, rmax = rates.min(), rates.max()
        for x, y, rate in zip(users['x'], users['y'], rates):
            i = min(int(x/env * self.grid_size), self.grid_size-1)
            j = min(int(y/env * self.grid_size), self.grid_size-1)
            norm = (rate-rmin)/(rmax-rmin) if rmax>rmin else 0.5
            grid[i, j] += 1 + norm

        # 4) Scale targets
        target = torch.tensor([
            hori[0]/env,
            hori[1]/env,
            alti/env
        ], dtype=torch.float32)
        env_scaled = torch.tensor(env/500, dtype=torch.float32)

        return (
            torch.from_numpy(grid).unsqueeze(0),  # (1, H, W)
            target,
            env_scaled,
            env  # keep original for any post‐processing
        )

# ─── DataLoaders ───────────────────────────────────────────────────────────────
dataset = UAVDataset(NUM_SAMPLES, GRID_SIZE, USER_DENSITY)
n_train = int(TRAIN_RATIO * NUM_SAMPLES)
n_val   = NUM_SAMPLES - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ─── Model ─────────────────────────────────────────────────────────────────────
class ResNetWithEnv(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        feats = base.fc.in_features
        base.fc = nn.Linear(feats, 128)
        self.backbone = base
        self.dropout  = nn.Dropout(DROP_RATE)
        self.env_fc   = nn.Linear(1, 32)
        self.head     = nn.Linear(128+32, 3)

    def forward(self, x, env_scaled):
        f = self.dropout(self.backbone(x))
        e = self.env_fc(env_scaled.unsqueeze(1))
        return self.head(torch.cat([f, e], dim=1))

model     = ResNetWithEnv().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE, verbose=True
)

# ─── Training Loop ────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    total_train = 0.0
    for grid, target, env_s, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        grid, target, env_s = grid.to(DEVICE), target.to(DEVICE), env_s.to(DEVICE)
        optimizer.zero_grad()
        preds = model(grid, env_s)
        loss  = criterion(preds, target)
        loss.backward()
        optimizer.step()
        total_train += loss.item()
    train_losses.append(total_train / len(train_loader))

    # Validate
    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for grid, target, env_s, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            grid, target, env_s = grid.to(DEVICE), target.to(DEVICE), env_s.to(DEVICE)
            total_val += criterion(model(grid, env_s), target).item()
    val_losses.append(total_val / len(val_loader))
    scheduler.step(val_losses[-1])

    print(f"Epoch {epoch:02d} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")

# ─── Save & Plot ──────────────────────────────────────────────────────────────
torch.save(model.state_dict(), f"cnn_res_{GRID_SIZE}_{NUM_SAMPLES}.pth")
import pandas as pd
pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}) \
  .to_csv(f"loss_{GRID_SIZE}.csv", index=False)

plt.plot(train_losses, label='Train')
plt.plot(val_losses,   label='Val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()
