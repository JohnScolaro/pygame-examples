"""
A collection of helpers used in multiple places in the puzzle grid.
"""

from typing import Any

from common.match3.player_actions import MoveAction


def apply_move_to_grid(
    grid: list[list[Any]], move_action: MoveAction
) -> list[list[Any]]:
    """
    Given any grid object (list of lists) mutates it to apply a move.
    """
    if move_action.row_or_col == "row":
        width = len(grid)
        amount = move_action.amount % width
        row = [grid[x][move_action.index] for x in range(width)]
        row = row[-amount:] + row[:-amount]
        for x in range(width):
            grid[x][move_action.index] = row[x]
    if move_action.row_or_col == "col":
        height = len(grid[0])
        amount = move_action.amount % height
        grid[move_action.index] = (
            grid[move_action.index][amount:] + grid[move_action.index][:amount]
        )
    return grid
