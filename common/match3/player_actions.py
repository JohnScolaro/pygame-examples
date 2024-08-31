import dataclasses
from typing import Literal


@dataclasses.dataclass
class MoveAction:
    row_or_col: Literal["row", "col"]
    index: int
    amount: int
