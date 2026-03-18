"""
Compare CLIP text embedding similarity structure between:
  - the original CLIP embedding space (dimension depends on the CLIP model, e.g., 512 for ViT-B-32)
  - a fixed random-projected space with dimension = --proj_dim (e.g., 256 / 128 / 64)

The projection mirrors this project's training-time projection:
  - Gaussian random matrix
  - QR to obtain roughly orthonormal columns
  - seeded for reproducibility
  - optional L2 normalization before/after projection

Typical usage (run multiple times to compare different proj dims):
  python tools/compare_clip_similarity_original_vs_projected.py --proj_dim 256 --seed 0
  python tools/compare_clip_similarity_original_vs_projected.py --proj_dim 128 --seed 0
  python tools/compare_clip_similarity_original_vs_projected.py --proj_dim 64  --seed 0

It prints:
  - cosine similarity matrix in the original space
  - cosine similarity matrix in the projected space
  - Pearson / Spearman agreement between the two matrices (upper triangle entries)
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + eps)


def fixed_random_projection_matrix(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    """
    Build a fixed random projection matrix W in R^{in_dim x out_dim}.
    We generate a Gaussian matrix then take Q from QR to make columns roughly orthonormal.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    a = torch.randn((in_dim, out_dim), generator=g, dtype=torch.float32)
    q = torch.linalg.qr(a, mode="reduced").Q  # [in_dim, out_dim]
    return q


def cosine_sim_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, D] assumed normalized.
    returns: [N, N] cosine similarity matrix.
    """
    return x @ x.t()


def upper_triangle_flat(m: torch.Tensor) -> np.ndarray:
    n = m.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    v = m[idx[0], idx[1]].detach().cpu().numpy()
    return v


def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Simple rankdata with average ranks for ties.
    Returns ranks starting at 1.
    """
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

    # Handle ties: average ranks within each tie group
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def format_matrix(m: np.ndarray, labels: List[str]) -> str:
    # Pretty-print with fixed width.
    w = max(len(s) for s in labels)
    header = " " * (w + 2) + " ".join([f"{i:>7d}" for i in range(len(labels))])
    lines = [header]
    for i, lab in enumerate(labels):
        row = " ".join([f"{m[i, j]:7.3f}" for j in range(len(labels))])
        lines.append(f"{lab:<{w}}  {row}")
    return "\n".join(lines)


def encode_text_open_clip(
    prompts: List[str],
    model_name: str,
    pretrained: str,
    device: str,
) -> torch.Tensor:
    try:
        import open_clip
    except Exception as e:
        raise ImportError(
            "open_clip_torch is required. Install it via requirements.txt."
        ) from e

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    args = ap.parse_args()

    # 4 actions x 2 prompt styles (short vs sentence), matching common training prompts.
    prompts = [
        "humanoid spinkick",
        "a humanoid performing a spinning kick",
        "humanoid roll",
        "a humanoid performing a roll",
        "humanoid backflip",
        "a humanoid performing a backflip",
        "humanoid run",
        "a humanoid running forward",
    ]

    x = encode_text_open_clip(prompts, args.model, args.pretrained, args.device).to(dtype=torch.float32)
    x = l2_normalize(x)
    d = int(x.shape[-1])

    if args.proj_dim == d:
        y = x
    else:
        w = fixed_random_projection_matrix(d, args.proj_dim, args.seed).to(device=x.device, dtype=x.dtype)
        y = x @ w
        y = l2_normalize(y)

    sim_orig = cosine_sim_matrix(x).detach().cpu().numpy()
    sim_proj = cosine_sim_matrix(y).detach().cpu().numpy()

    vorig = upper_triangle_flat(torch.as_tensor(sim_orig))
    vproj = upper_triangle_flat(torch.as_tensor(sim_proj))
    pearson = float(np.corrcoef(vorig, vproj)[0, 1])
    spear = spearmanr(vorig, vproj)

    print(f"CLIP model={args.model} pretrained={args.pretrained} device={args.device}")
    print(f"orig_dim={d} proj_dim={args.proj_dim} seed={args.seed}")
    print()
    labels = [f"{i}:{p[:28]}{'...' if len(p) > 28 else ''}" for i, p in enumerate(prompts)]
    print("Cosine similarity matrix (original space):")
    print(format_matrix(sim_orig, labels))
    print()
    print("Cosine similarity matrix (projected space):")
    print(format_matrix(sim_proj, labels))
    print()
    print("Similarity-structure agreement (upper triangle):")
    print(f"  Pearson r   = {pearson:.4f}")
    print(f"  Spearman ρ  = {spear:.4f}")


if __name__ == "__main__":
    main()

