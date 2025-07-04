import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from minesweeper_env import MinesweeperEnv
from mc_agent import MCAgent
from rotate_utils import rotate_board, rotate_action

# Make sure the output directory for graphs exists
os.makedirs("graphs", exist_ok=True)

EPISODES = 100000
GAMMA = 0.99

# Initialize environment and agent
env = MinesweeperEnv(width=5, height=5, n_mines=3)
agent = MCAgent(gamma=GAMMA)

# Load pretrained model if available
if os.path.exists("pretrained_supervised.pt"):
    agent.model.load_state_dict(torch.load("pretrained_supervised.pt"))
    print("Loaded pretrained_supervised.pt")
elif os.path.exists("mc_final.pt"):
    agent.model.load_state_dict(torch.load("mc_final.pt"))
    print("Loaded mc_final.pt")

# History logs
reward_history = []
win_history = []
step_history = []

# Helper to flatten the board
def flatten(b): return b.flatten().astype(np.float32)

# Training loop
for ep in range(1, EPISODES+1):
    s = env.reset()
    prev_flat = flatten(s)
    done = False
    states, actions, rewards = [], [], []
    step_count = 0

    while not done:
        curr_flat = flatten(s)
        inp = np.concatenate([prev_flat, curr_flat])  # Use previous and current state as input
        q = agent.predict(inp)

        # Mask out already revealed cells
        mask = (curr_flat == -2)
        q[~mask] = -np.inf

        # Compute softmax probabilities with numerical stability
        expq = np.exp(q - np.nanmax(q))
        expq[~np.isfinite(expq)] = 0
        p = expq / (expq.sum() if expq.sum() > 0 else 1)

        # Sample action based on probability distribution
        a = np.random.choice(len(p), p=p)

        s2, r, done, _ = env.step(a)
        step_count += 1

        # Adjust reward based on outcome
        if done and not env._check_win():
            r2 = -5.0  # Penalty for failure
        elif done and env._check_win():
            r2 = 10.0  # Reward for success
        else:
            r2 = -0.01  # Small penalty to encourage quicker decisions

        states.append(np.concatenate([prev_flat, curr_flat]))
        actions.append(a)
        rewards.append(r2)

        # Data augmentation using board rotation
        for k in range(1, 4):
            p_rot = rotate_board(prev_flat, k)
            c_rot = rotate_board(curr_flat, k)
            a_rot = rotate_action(a, k)
            states.append(np.concatenate([p_rot, c_rot]))
            actions.append(a_rot)
            rewards.append(r2)

        prev_flat = curr_flat
        s = s2

    step_history.append(step_count)

    # Compute discounted returns
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    episode = list(zip(states, actions, returns))

    # Train agent only on successful episodes
    if env._check_win():
        agent.update(episode)

    reward_history.append(sum(rewards))
    win_history.append(1 if env._check_win() else 0)

    if ep % 100 == 0:
        avg_win = sum(win_history[-100:]) / 100
        print(f"Ep {ep} | Win Rate: {avg_win*100:.2f}% | Total Wins: {sum(win_history)}")

# Save model after training
torch.save(agent.model.state_dict(), "mc_final.pt")
print("Training complete.")

# ========== Graphs ========== #
window = 100
rolling_reward = np.convolve(reward_history, np.ones(window) / window, mode='valid')
rolling_win = np.convolve(win_history, np.ones(window) / window, mode='valid')

def save_and_show(fig_name):
    plt.tight_layout()
    path = os.path.join("graphs", fig_name)
    plt.savefig(path)
    plt.show()

# Plot 1: Rolling mean reward
plt.figure(figsize=(6, 4))
plt.plot(rolling_reward)
plt.title(f"Rolling Mean Reward ({window})")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid()
save_and_show("rolling_reward.png")

# Plot 2: Rolling win rate
plt.figure(figsize=(6, 4))
plt.plot(rolling_win)
plt.title(f"Win Rate ({window})")
plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.grid()
save_and_show("rolling_win_rate.png")

# Plot 3: Steps in failed episodes
failure_steps = [s for s, win in zip(step_history, win_history) if not win]
plt.figure(figsize=(6, 4))
plt.plot(failure_steps)
plt.title("Steps Until Failure (Only Failed Episodes)")
plt.xlabel("Failure Episode Index")
plt.ylabel("Steps")
plt.grid()
save_and_show("steps_until_failure.png")

# Plot 4: Steps in successful episodes
success_steps = [s for s, win in zip(step_history, win_history) if win]
plt.figure(figsize=(6, 4))
plt.plot(success_steps)
plt.title("Steps Until Victory (Only Successful Episodes)")
plt.xlabel("Victory Episode Index")
plt.ylabel("Steps")
plt.grid()
save_and_show("steps_until_victory.png")

# Plot 5: Compare success rate: first vs last 100 episodes
first_100_success = sum(win_history[:100]) / 100
last_100_success = sum(win_history[-100:]) / 100
plt.figure(figsize=(6, 4))
plt.bar(['First 100', 'Last 100'], [first_100_success, last_100_success])
plt.title("Success Comparison: First vs Last 100 Episodes")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
save_and_show("success_comparison.png")

# Plot 6: Cumulative win rate
cumulative_win_rate = np.cumsum(win_history) / (np.arange(len(win_history)) + 1)
plt.figure(figsize=(6, 4))
plt.plot(cumulative_win_rate, color='green')
plt.title("Cumulative Win Rate Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Cumulative Win Rate")
plt.ylim(0, 1)
plt.grid()
save_and_show("cumulative_win_rate.png")
