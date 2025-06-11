import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionTopK(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_classes=3, top_k=70, dropout=0.3):
        super(AttentionTopK, self).__init__()
        self.top_k = top_k

        # Attention mechanism
        self.attention_V = nn.Linear(L, D)
        self.attention_U = nn.Linear(L, D)
        self.attention_weights = nn.Linear(D, K)

        # Dropout after attention
        self.attention_dropout = nn.Dropout(dropout)

        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(L, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: [B, N, L] = (batch size, #patches, feature dim)
        mask: [B, N] binary mask indicating valid patches
        """
        A_V = torch.tanh(self.attention_V(x))           # (B, N, D)
        A_U = torch.sigmoid(self.attention_U(x))        # (B, N, D)
        A = A_V * A_U                                   # (B, N, D)
        A = self.attention_weights(A)                   # (B, N, K)
        A = torch.softmax(A, dim=1)                     # Attention weights
        A = A.squeeze(-1)                               # (B, N)

        if mask is not None:
            A = A * mask
            A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

        A = self.attention_dropout(A)  # Apply dropout after attention

        # Safe Top-K Patch Selection
        k = min(self.top_k, x.size(1))
        topk_vals, topk_idxs = torch.topk(A, k, dim=1)
        topk_features = torch.gather(x, 1, topk_idxs.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # (B, k, L)

        pooled = topk_features.mean(dim=1)              # Mean pooling
        Y_prob = self.classifier(pooled)                # Class logits
        Y_hat = torch.argmax(Y_prob, dim=1)

        return Y_prob, Y_hat, A

    def calculate_objective(self, x, y, mask=None, loss_fn=None):
        y = y.long()
        Y_prob, _, _ = self.forward(x, mask=mask)
        loss = loss_fn(Y_prob, y)
        return loss, Y_prob, y

