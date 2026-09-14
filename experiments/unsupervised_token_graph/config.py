"""Configuration for the unsupervised token-graph pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    attention_floor: float = 0.05
    hidden_dim: int = 64
    message_steps: int = 2
    epochs: int = 20
    learning_rate: float = 1e-3
    reference_fraction: float = 0.8
    alarm_budget: float = 0.05
    seed: int = 20260914
