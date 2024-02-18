import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pooler = Pooler(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc(self.pooler(x))
        return x


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_out = self.dense(hidden_states)
        pooled_out = self.activation(pooled_out)
        return pooled_out


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.pooler = Pooler(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            self.pooler,
            self.fc1,
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class MAEHead(nn.Module):
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.decoder(x)       
        return x


class OneHotCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, one_hot_labels):
        log_prob = F.log_softmax(logits, dim=-1)
        loss = (-one_hot_labels * log_prob).sum(dim=-1).mean()
        return loss


