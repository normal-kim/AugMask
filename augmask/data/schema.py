"""Column roles of a cleaned dataset."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

TASKS = ("binclass", "multiclass", "regression")


@dataclass(frozen=True)
class Schema:
    """Roles of the columns of a complete (cleaned) table.

    ``columns`` is the canonical column order of the data-space table; missingness
    masks are stored in this order. Numerical columns are float, categorical
    columns (and a classification target) are strings.
    """

    name: str
    task: str
    columns: tuple[str, ...]
    num_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]
    target: str

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        roles = list(self.num_cols) + list(self.cat_cols) + [self.target]
        if sorted(roles) != sorted(self.columns) or len(set(roles)) != len(roles):
            raise ValueError("num_cols, cat_cols and target must partition columns")

    @property
    def is_regression(self) -> bool:
        return self.task == "regression"

    @property
    def num_block(self) -> tuple[str, ...]:
        """Numerical block as fed to the models; a regression target comes first."""
        return ((self.target,) if self.is_regression else ()) + self.num_cols

    @property
    def cat_block(self) -> tuple[str, ...]:
        """Categorical block as fed to the models; a classification target comes first."""
        return (() if self.is_regression else (self.target,)) + self.cat_cols

    def block_positions(self) -> tuple[list[int], list[int]]:
        """Positions of the numerical and categorical blocks in ``columns``."""
        pos = {c: i for i, c in enumerate(self.columns)}
        return [pos[c] for c in self.num_block], [pos[c] for c in self.cat_block]

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path) -> "Schema":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
