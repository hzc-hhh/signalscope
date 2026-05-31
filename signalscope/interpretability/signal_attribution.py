"""
Signal attribution analysis.

Tools for understanding which parts of an input signal drive model predictions.
"""


import numpy as np
import torch


def attention_weights(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_name: str | None = None,
) -> np.ndarray:
    """
    Extract attention weights from a transformer model.

    Parameters
    ----------
    model : nn.Module
        Transformer model with attention layers.
    x : torch.Tensor
        Input signal.
    layer_name : str, optional
        Specific attention layer to extract from.

    Returns
    -------
    np.ndarray
        Attention weight matrix.
    """
    model.eval()

    def hook_fn(module, input_, output):
        # Some attention implementations return weights as second output
        pass

    hooks = []
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()

    return np.array([])  # Simplified; real impl would capture and aggregate


def gradient_attribution(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: int | None = None,
) -> np.ndarray:
    """
    Compute input-gradient attribution (saliency map for 1D signals).

    Parameters
    ----------
    model : nn.Module
        Trained model.
    x : torch.Tensor, shape (B, L) or (1, L)
        Input signal.
    target : int, optional
        Target class index.

    Returns
    -------
    np.ndarray
        Attribution map matching input shape.
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    if x.dim() == 1:
        x = x.unsqueeze(0)

    output = model(x)
    if target is None:
        target = output.argmax(dim=-1).item() if output.dim() > 1 else 0

    if output.dim() > 1:
        score = output[0, target]
    else:
        score = output.sum()

    model.zero_grad()
    score.backward()

    attribution = x.grad.abs().cpu().numpy()
    return attribution.squeeze()
