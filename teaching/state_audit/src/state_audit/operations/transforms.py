"""Delete, replace, inject and steer a selected block of a representation."""

from dataclasses import dataclass

import torch

from .target import Target


@dataclass
class Delete:
    target: Target
    strength: float = 1.0

    def apply(self, selected):
        return selected * (1 - self.strength)


@dataclass
class Replace:
    target: Target
    value: object

    def apply(self, selected):
        return torch.as_tensor(self.value, device=selected.device, dtype=selected.dtype)


@dataclass
class Inject:
    target: Target
    value: object
    scale: float = 1.0

    def apply(self, selected):
        value = torch.as_tensor(self.value, device=selected.device, dtype=selected.dtype)
        return selected + self.scale * value


@dataclass
class Steer:
    """Add amount × unit direction along the final feature axis."""

    target: Target
    direction: object
    amount: float = 1.0

    def __post_init__(self):
        if self.target.representation == "attention":
            raise ValueError("Steer uses a feature direction; use edge operations for attention")
        direction = torch.as_tensor(self.direction, dtype=torch.float32)
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(norm).all() or (norm == 0).any():
            raise ValueError("A steering direction must have a finite, nonzero norm")
        self.direction = direction / norm

    def apply(self, selected):
        direction = self.direction.to(device=selected.device, dtype=selected.dtype)
        return selected + self.amount * direction


def apply_operations(tensor, operations):
    """Clone once; operations at the same site execute in the supplied order."""
    changed = tensor.clone()
    for operation in operations:
        indices = operation.target.indices(changed)
        changed[indices] = operation.apply(changed[indices])
    return changed
