"""PyTorch dataset and collation helpers for MIL bags."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from datasets.daic import BagRecord


@dataclass(frozen=True)
class BagBatch:
    """A mini-batch of variable-length participant bags."""

    records: list[BagRecord]

    @property
    def labels(self) -> torch.Tensor:
        """Return bag labels as a float tensor."""
        return torch.tensor([record.label for record in self.records], dtype=torch.float32)

    @property
    def bag_lengths(self) -> list[int]:
        """Return instance counts for each bag."""
        return [len(record.instances) for record in self.records]

    @property
    def participant_ids(self) -> list[int]:
        """Return participant IDs in batch order."""
        return [record.participant_id for record in self.records]

    @property
    def flat_texts(self) -> list[str]:
        """Return all instance texts flattened in bag order."""
        return [text for record in self.records for text in record.instances]


class TextBagDataset(Dataset[BagRecord]):
    """Dataset of participant-level MIL bags."""

    def __init__(self, records: list[BagRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BagRecord:
        return self.records[index]


def collate_bags(records: list[BagRecord]) -> BagBatch:
    """Collate participant bags without padding instances."""
    return BagBatch(records=records)

