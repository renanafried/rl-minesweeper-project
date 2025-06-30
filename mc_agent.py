import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, output_size=25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)

class MCAgent:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.model = FullyConnectedNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.model(t).squeeze(0).numpy()
        return q

    def update(self, episode):
        self.model.train()
        states, actions, returns = zip(*episode)
        S = torch.tensor(states, dtype=torch.float32)
        A = torch.tensor(actions, dtype=torch.long)
        G = torch.tensor(returns, dtype=torch.float32)

        logits = self.model(S)
        logp = F.log_softmax(logits, dim=1)
        sel = logp[range(len(A)), A]
        loss = -(sel * G).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
