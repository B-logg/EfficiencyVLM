"""
Norm-based visual token pruner (FastV-style, pre-LLM).

After MLP1 projection, each visual token has embedding in LLM space.
Tokens with lower L2 norm are considered less important and pruned.
Pruning before LLM input reduces LLM sequence length → faster attention.

Keep ratio 50%: 256 tokens → 128 tokens
"""
import time
import torch


class NormBasedTokenPruner:
    def __init__(self, keep_ratio: float = 0.5):
        assert 0.0 < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
        self.keep_ratio = keep_ratio

    @torch.no_grad()
    def prune(self, img_embs: torch.Tensor):
        """
        Args:
            img_embs: [N, D] visual token embeddings after MLP1
        Returns:
            pruned_embs: [N_keep, D]
            t_ms: wall-clock pruning time in milliseconds
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        N = img_embs.shape[0]
        n_keep = max(1, int(N * self.keep_ratio))

        importance = img_embs.norm(dim=-1)          # [N]
        keep_idx = importance.topk(n_keep).indices.sort().values  # [N_keep]
        pruned = img_embs[keep_idx]                 # [N_keep, D]

        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0

        return pruned, t_ms
