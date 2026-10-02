"""
heads.py — Classification Head Utilities

Helper utilities for the multi-label classification head.
The actual head is defined inline in hybrid_model.py; this module provides
diagnostic and inspection tools.
"""

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """Count total (or trainable-only) parameters in a model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_summary(model: nn.Module) -> dict[str, int]:
    """Return a per-submodule parameter count."""
    summary = {}
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        summary[name] = {"total": total, "trainable": trainable}
    return summary


def verify_no_softmax(model: nn.Module) -> bool:
    """Verify that no Softmax layer is present in the model.

    This is a multi-label problem — each output is an independent logit.
    Softmax would incorrectly impose competition between classes.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Softmax, nn.LogSoftmax)):
            print(f"[heads] WARNING: Found {type(module).__name__} at '{name}' — "
                  "this should NOT be present in a multi-label model!")
            return False
    return True


def test_model_output_shape(
    model: nn.Module,
    batch_size: int = 2,
    num_classes: int = 14,
    image_size: int = 224,
    device: str = "cpu",
) -> bool:
    """Unit test: verify the model produces the expected output shape."""
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch_size, num_classes)
    if output.shape != expected_shape:
        print(f"[heads] FAIL: expected shape {expected_shape}, got {output.shape}")
        return False

    print(f"[heads] ✓ Output shape: {output.shape}")
    return True
