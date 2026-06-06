import torch
from torch import nn


class Model_28(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.dense1 = nn.Linear(128*8*4, 256)
        self.policy_head = nn.Linear(256, 33)
        self.value_head = nn.Linear(256, 1)
        self.bid_head = nn.Linear(256, 20)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    # Playing mask is one for legal moves and 0 for illegal moves
    def forward(self, x, playing_mask=None, bid_mask=None):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)

        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)

        policy_logits = self.policy_head(x)
        if playing_mask is not None:
            policy_logits = policy_logits + ((playing_mask-1) * 1e9)
        policy_outputs = self.softmax(policy_logits)

        value_output = self.value_head(x)
        value_output = self.tanh(value_output)

        bid_logits = self.bid_head(x)
        if bid_mask is not None:
            bid_logits = bid_logits + ((bid_mask-1) * 1e9)
        bid_outputs = self.softmax(bid_logits)

        return policy_outputs, value_output, bid_outputs



# Policy head's output -> 32 for each card. last one whether to open trump or not
# Value head -> whether we will win or not
# bid_head -> first