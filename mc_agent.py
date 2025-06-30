import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, output_size=25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),      # First hidden layer
            nn.LayerNorm(hidden_size),               # Normalize layer output
            nn.LeakyReLU(),                           # Activation function
            nn.Linear(hidden_size, hidden_size),     # Second hidden layer
            nn.LeakyReLU(),                           # Activation function
            nn.Linear(hidden_size, output_size)      # Output layer
        )

    def forward(self, x):
        return self.model(x)  # Forward pass

class MCAgent:
    def __init__(self, gamma=0.99):
        self.gamma = gamma  # Discount factor
        self.model = FullyConnectedNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            q = self.model(t).squeeze(0).numpy()                       # Remove batch dimension
        return q  # Return predicted Q-values

    def update(self, episode):
        self.model.train()
        states, actions, returns = zip(*episode)

        S = torch.tensor(states, dtype=torch.float32)
        A = torch.tensor(actions, dtype=torch.long)
        G = torch.tensor(returns, dtype=torch.float32)

        logits = self.model(S)                      # Forward pass
        logp = F.log_softmax(logits, dim=1)         # Compute log-probabilities
        sel = logp[range(len(A)), A]                # Select log-prob for taken actions
        loss = -(sel * G).mean()                    # Policy gradient loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
