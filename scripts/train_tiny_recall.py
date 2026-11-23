"""Regenerate the demo checkpoint: uv run --with torch==2.15.0 scripts/train_tiny_recall.py."""
import itertools
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TOKENS = ("A", "B", "C", "1", "2", "3", "?")
WIDTH = 16
HEADS = 2
STEPS = 1600


def parameter(inputs, outputs):
    matrix = torch.empty(inputs, outputs)
    nn.init.xavier_uniform_(matrix)
    return nn.Parameter(matrix)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q", "k", "v", "out"):
            setattr(self, name, parameter(WIDTH, WIDTH))
        self.up = parameter(WIDTH, WIDTH * 2)
        self.down = parameter(WIDTH * 2, WIDTH)

    def forward(self, x):
        batch, length, _ = x.shape
        n = F.layer_norm(x, (WIDTH,))

        def split(matrix):
            return (n @ matrix).reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)

        q, k, v = map(split, (self.q, self.k, self.v))
        scores = q @ k.transpose(-2, -1) / (WIDTH // HEADS) ** 0.5
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        context = scores.masked_fill(mask, -torch.inf).softmax(dim=-1) @ v
        residual = x + context.transpose(1, 2).reshape(batch, length, WIDTH) @ self.out
        mlp = (F.layer_norm(residual, (WIDTH,)) @ self.up).relu() @ self.down
        return residual + mlp


class TinyRecall(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.randn(len(TOKENS), WIDTH) * 0.18)
        self.position = nn.Parameter(torch.randn(8, WIDTH) * 0.18)
        self.blocks = nn.ModuleList([DecoderBlock(), DecoderBlock()])
        self.readout = parameter(WIDTH, len(TOKENS))

    def forward(self, ids):
        x = self.token[ids] + self.position[:ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return F.layer_norm(x, (WIDTH,)) @ self.readout


def main():
    torch.manual_seed(8249)
    torch.set_num_threads(1)
    examples = []
    for keys in itertools.permutations(range(3)):
        for values in itertools.product(range(3, 6), repeat=3):
            for query in range(3):
                pairs = [item for key in keys for item in (key, values[key])]
                examples.append((pairs + [6, query], values[query]))
    ids = torch.tensor([prompt for prompt, _ in examples])
    targets = torch.tensor([answer for _, answer in examples])
    # Hold out whole contexts so the test set does not share prompts with a different final key.
    order = torch.randperm(len(examples) // 3)
    train, test = [torch.stack([group * 3 + i for i in range(3)], dim=1).flatten()
                   for group in (order[16:], order[:16])]

    model = TinyRecall()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    for _ in range(STEPS):
        batch = train[torch.randint(len(train), (96,))]
        loss = F.cross_entropy(model(ids[batch])[:, -1], targets[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        correct = int((model(ids[test])[:, -1].argmax(dim=-1) == targets[test]).sum())
        fixture = torch.tensor([[0, 4, 1, 3, 2, 5, 6, 0]])
        checkpoint = {
            "weights": {
                "token": model.token.tolist(),
                "position": model.position.tolist(),
                "readout": model.readout.tolist(),
                "blocks": [
                    {name: tensor.tolist() for name, tensor in block.state_dict().items()}
                    for block in model.blocks
                ],
            },
            "evaluation": {"correct": correct, "total": len(test)},
            "referenceFinalLogits": model(fixture)[0, -1].tolist(),
        }

    output = Path(__file__).resolve().parents[1] / "frontend/src/engine/inference/tinyRecall.weights.json"
    output.write_text(json.dumps(checkpoint, separators=(",", ":")) + "\n")
    print(f"Held-out prompts: {correct}/{len(test)} correct · {sum(p.numel() for p in model.parameters())} parameters")


if __name__ == "__main__":
    main()
