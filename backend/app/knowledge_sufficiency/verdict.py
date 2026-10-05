"""What a sufficiency check answers with."""

from enum import Enum
from typing import List, NamedTuple

from numpy import float32
from numpy.typing import NDArray


class Sufficiency(str, Enum):
    """How much of the answer the supplied knowledge already holds."""

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class Verdict(NamedTuple):
    sufficiency: Sufficiency
    best: float
    supporting: List
    scores: NDArray[float32]

    @property
    def is_sufficient(self) -> bool:
        return self.sufficiency is Sufficiency.FULL

    @property
    def needs_retrieval(self) -> bool:
        return self.sufficiency is not Sufficiency.FULL
