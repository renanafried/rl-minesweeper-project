import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from minesweeper_env import MinesweeperEnv
from mc_agent import FullyConnectedNet
from rotate_utils import rotate_board, rotate_action
from solver import get_safe_moves

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 20000
BATCH        = 64
EPOCHS       = 90

X, y = [], []
env = MinesweeperEnv(width=5, height=5, n_mines=3)

# Data generation loop
for ep in range(NUM_EPISODES):
    vis = env.reset()
    done = False

    # 1 random click to open initial region
    unopened = np.where(vis.flatten() == -2)[0]
    if len(unopened):
        a0, = random.sample(list(unopened), 1)
        vis, _, done, _ = env.step(a0)
    if done:
        continue

    flat1 = vis.flatten().astype(np.float32)

    # Up to 3 more random steps to reach a meaningful state
    for _ in range(3):
        unopened = np.where(vis.flatten() == -2)[0]
        if not len(unopened):
            break
        a = random.choice(unopened)
        vis, r, done, _ = env.step(a)
        if r < 0 or done:
            break
    if done:
        continue

    flat2 = vis.flatten().astype(np.float32)

    # Try solving the current state logically
    moves = get_safe_moves(vis)

    # Fallback if solver fails: pick all safe unopened cells
    if not moves:
        flat_board = env.board.flatten()
        for idx in np.where(vis.flatten() == -2)[0]:
            if flat_board[idx] != -1:
                i, j = divmod(int(idx), env.w)
                moves.add((i, j))

    added = 0
    for (i, j) in moves:
        action = i * env.w + j
        X.append(np.concatenate([flat1, flat2]))
        y.append(action)

        # Add rotated versions for data augmentation
        for k in (1, 2, 3):
            f1r = rotate_board(flat1, k)
            f2r = rotate_board(flat2, k)
            ar  = rotate_action(action, k)
            X.append(np.concatenate([f1r, f2r]))
            y.append(ar)
        added += 1

    if ep % 100 == 0:
        print(f"[Ep {ep}] Added {added} logical moves")

# Save data
if not X:
    print("No data generated. Solver failed.")
    exit()

X = np.array(X);  y = np.array(y)
np.savez("solver_data.npz", X=X, y=y)
print(f"Saved solver_data.npz with {len(X)} examples.")

# ========== Supervised pretraining ========== #
print("Starting supervised pretraining…")

X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y, dtype=torch.long,    device=DEVICE)

model = FullyConnectedNet().to(DEVICE)
opt   = optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    perm = torch.randperm(X_t.size(0), device=DEVICE)
    Xt   = X_t[perm];  yt = y_t[perm]
    epoch_loss = 0.0
    for i in range(0, len(Xt), BATCH):
        xb = Xt[i:i+BATCH];  yb = yt[i:i+BATCH]
        opt.zero_grad()
        out = model(xb)
        loss = lossf(out, yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} – Loss: {epoch_loss / (len(Xt)//BATCH):.4f}")

# Save trained model
torch.save(model.state_dict(), "pretrained_supervised.pt")
print("Supervised pretraining complete.")
