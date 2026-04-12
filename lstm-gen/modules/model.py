import torch
import torch.nn as nn


class CharLM(nn.Module):
    """
    Character-level LSTM language model.

    Input : (B, T) token indices
    Output: (B, T, vocab_size) logits (one per timestep for next-char prediction)

    During generation the hidden state is threaded across calls so the model
    can produce arbitrarily long sequences one token at a time.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb    = self.embed(x)           # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        logits = self.head(self.drop(out))    # (B, T, V)
        return logits, hidden

    @torch.no_grad()
    def generate(self, prompt_idx: list[int], max_new_tokens: int,
                 temperature: float = 1.0, device: str = "cpu") -> list[int]:
        self.eval()

        # Warm up hidden state with the full prompt in one forward pass.
        x = torch.tensor([prompt_idx], dtype=torch.long, device=device)  # (1, T_prompt)
        logits, hidden = self(x)

        generated = list(prompt_idx)

        # Sample the first new token from the last prompt position.
        next_tok = self._sample(logits[:, -1, :], temperature)
        generated.append(next_tok)

        # Generate remaining tokens one at a time, threading hidden state.
        for _ in range(max_new_tokens - 1):
            x = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, hidden = self(x, hidden)
            next_tok = self._sample(logits[:, -1, :], temperature)
            generated.append(next_tok)

        return generated

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
