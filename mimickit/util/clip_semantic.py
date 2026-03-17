import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

import util.mp_util as mp_util


@dataclass(frozen=True)
class ClipTextSemanticConfig:
    enabled: bool
    prompt: str
    model_name: str
    pretrained: str
    proj_dim: int
    cache_path: str
    l2_normalize: bool = True
    proj_seed: int = 0


def parse_clip_text_semantic_config(env_config: dict) -> ClipTextSemanticConfig:
    cfg = env_config.get("global_semantic", {}) or {}
    sem_type = cfg.get("type", "")
    enabled = bool(cfg.get("enabled", False))
    if sem_type not in ("", "clip_text"):
        raise ValueError(f"Unsupported global_semantic.type={sem_type!r}. Expected 'clip_text'.")

    prompt = str(cfg.get("prompt", "")).strip()
    model_name = str(cfg.get("model_name", "ViT-B-32"))
    pretrained = str(cfg.get("pretrained", "openai"))
    proj_dim = int(cfg.get("proj_dim", 256))
    cache_path = str(cfg.get("cache_path", "")).strip()
    l2_normalize = bool(cfg.get("l2_normalize", True))
    proj_seed = int(cfg.get("proj_seed", 0))

    if enabled and not prompt:
        raise ValueError("global_semantic is enabled but prompt is empty.")
    if enabled and proj_dim <= 0:
        raise ValueError(f"proj_dim must be > 0, got {proj_dim}.")

    return ClipTextSemanticConfig(
        enabled=enabled,
        prompt=prompt,
        model_name=model_name,
        pretrained=pretrained,
        proj_dim=proj_dim,
        cache_path=cache_path,
        l2_normalize=l2_normalize,
        proj_seed=proj_seed,
    )


def _load_cached_vec(cache_path: str, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if not cache_path:
        return None
    if not os.path.exists(cache_path):
        return None
    vec = np.load(cache_path)
    vec = torch.as_tensor(vec, device=device, dtype=dtype)
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)  # [1, D]
    return vec


def _maybe_save_cache(cache_path: str, vec_1d: np.ndarray) -> None:
    if not cache_path:
        return
    out_dir = os.path.dirname(cache_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(cache_path, vec_1d)


def _compute_clip_text_embedding(
    prompt: str,
    model_name: str,
    pretrained: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        import open_clip
    except Exception as e:
        raise ImportError(
            "open_clip_torch is required for CLIP global semantic. "
            "Install dependencies from requirements.txt."
        ) from e

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device=device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)
    tokens = tokenizer([prompt]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features.to(dtype=dtype)
    return text_features  # [1, D]


def _fixed_random_projection(in_dim: int, out_dim: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    w = torch.randn((in_dim, out_dim), generator=g, dtype=torch.float32)
    w = torch.linalg.qr(w, mode="reduced").Q  # [in_dim, out_dim], roughly orthonormal columns
    w = w.to(device=device, dtype=dtype)
    return w


def build_global_semantic_vector(env_config: dict, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      - sem_vec: [1, proj_dim] tensor (or [1, 0] if disabled)
      - sem_dim: proj_dim (or 0 if disabled)
    """
    cfg = parse_clip_text_semantic_config(env_config)
    if not cfg.enabled:
        empty = torch.zeros((1, 0), device=device, dtype=dtype)
        return empty, 0

    cached = _load_cached_vec(cfg.cache_path, device=device, dtype=dtype)
    if cached is not None:
        sem_vec = cached
    else:
        # Compute on CPU by default to avoid GPU memory spikes, then move.
        clip_device = torch.device("cpu")
        clip_dtype = torch.float32
        clip_vec = _compute_clip_text_embedding(
            prompt=cfg.prompt,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            device=clip_device,
            dtype=clip_dtype,
        )  # [1, D]

        if cfg.l2_normalize:
            clip_vec = torch.nn.functional.normalize(clip_vec, p=2, dim=-1)

        clip_dim = int(clip_vec.shape[-1])
        if cfg.proj_dim == clip_dim:
            proj_vec = clip_vec
        else:
            w = _fixed_random_projection(clip_dim, cfg.proj_dim, cfg.proj_seed, device=clip_device, dtype=clip_dtype)
            proj_vec = clip_vec @ w  # [1, proj_dim]
            if cfg.l2_normalize:
                proj_vec = torch.nn.functional.normalize(proj_vec, p=2, dim=-1)

        # Save cache once (best-effort).
        if mp_util.is_root_proc():
            try:
                _maybe_save_cache(cfg.cache_path, proj_vec.squeeze(0).cpu().numpy())
            except Exception:
                pass

        sem_vec = proj_vec.to(device=device, dtype=dtype)

    # Ensure [1, D]
    if sem_vec.ndim == 1:
        sem_vec = sem_vec.unsqueeze(0)
    sem_dim = int(sem_vec.shape[-1])
    return sem_vec, sem_dim

