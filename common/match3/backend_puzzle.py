import dataclasses
import random
from collections import Counter
from copy import deepcopy

from common.match3.gem_types import GemTypes
from common.match3.helpers import apply_move_to_grid
from common.match3.player_actions import MoveAction

Match = list[tuple[int, int]]


@dataclasses.dataclass
class BackendGem:
    gem_type: GemTypes


@dataclasses.dataclass
class ExplodeAndReplacePhase:
    """
    A list of matches and the replacements to fill the gaps created by
    exploding the matches.
    """

    matches: list[Match]
    replacements: list[tuple[int, list[GemTypes]]]

    def is_nothing_to_do(self) -> bool:
        return not bool(self.matches)


class BackendPuzzleState:
    """
    The backend puzzle state. Reads player actions, and responds with backend
    actions to display to the player via the frontend puzzle.
    """

    def __init__(self, width: int, height: int) -> None:
        self.puzzle_state = self.get_initial_puzzle_state_with_no_matches(width, height)
        self.width = width
        self.height = height
        self.next_gems_to_spawn: list[GemTypes] = []

    def get_initial_puzzle_state_with_no_matches(
        self, width: int, height: int
    ) -> list[list[BackendGem]]:
        """
        Creates a 2D array of gem types with no more than 2 consecutive gems of
        the same color in any row or column.

        Args:
          width: Width of the grid (number of columns).
          length: Length of the grid (number of rows).

        Returns:
          A 2D list containing the gem types.
        """
        # Create a list of all gem types
        gem_types = set(gem_type for gem_type in GemTypes)

        # Initialize the grid with empty lists for each row
        grid: list[list[BackendGem]] = [[] for _ in range(width)]

        for x in range(width):
            for y in range(height):

                possible_gems = gem_types

                if y >= 2:
                    if grid[x][y - 1].gem_type == grid[x][y - 2].gem_type:
                        possible_gems = possible_gems - set([grid[x][y - 1].gem_type])

                if x >= 2:
                    if grid[x - 1][y].gem_type == grid[x - 2][y].gem_type:
                        possible_gems = possible_gems - set([grid[x - 1][y].gem_type])

                grid[x].append(BackendGem(gem_type=random.choice(list(possible_gems))))

        return grid

    def get_next_explode_and_replace_phase(
        self, actions: list[MoveAction]
    ) -> ExplodeAndReplacePhase:
        """
        Given the following list of actions, get the next phase of exploding and replacing.
        """
        # Apply actions
        for action in actions:
            self.puzzle_state = self._get_puzzle_state_after_move(
                self.puzzle_state, action
            )

        # Get matches
        matches = self._get_matches(self.puzzle_state)

        replacements = []
        counter = Counter(x for x, y in set(gem for gems in matches for gem in gems))
        for x in range(self.width):
            replacements_for_col = []
            for _ in range(counter[x]):
                replacements_for_col.append(self._get_next_gem_to_spawn().gem_type)
            if replacements_for_col:
                replacements.append((x, replacements_for_col))

        # Sort replacements from left to right
        replacements.sort(key=lambda x: x[0])

        # Apply this phase to the current state
        explode_and_replace_phase = ExplodeAndReplacePhase(
            matches=matches, replacements=replacements
        )
        self.apply_explode_and_replace_phase(explode_and_replace_phase)

        # Return the applied phase
        return explode_and_replace_phase

    def get_matches_from_hypothetical_move(
        self, move_action: MoveAction
    ) -> list[Match]:
        hypothetical_state = self._get_puzzle_state_after_move(
            self.puzzle_state, move_action
        )
        return self._get_matches(hypothetical_state)

    def _get_next_gem_to_spawn(self) -> BackendGem:
        if self.next_gems_to_spawn:
            return BackendGem(gem_type=self.next_gems_to_spawn.pop(0))
        return BackendGem(gem_type=random.choice(list(GemTypes)))

    def add_next_gem_to_spawn(self, gem_type: GemTypes) -> None:
        self.next_gems_to_spawn.append(gem_type)

    def add_next_gems_to_spawn(self, gem_types: list[GemTypes]) -> None:
        self.next_gems_to_spawn.extend(gem_types)

    @staticmethod
    def _get_matches(puzzle_state: list[list[BackendGem]]) -> list[Match]:
        matches = []

        width = len(puzzle_state)
        height = len(puzzle_state[0])

        # Get all vertical matches
        for x in range(width):
            match = []
            last = puzzle_state[x][0]
            match.append((x, 0))
            for y in range(1, height):
                if puzzle_state[x][y] == last:
                    match.append((x, y))
                else:
                    if len(match) > 2:
                        matches.append(match)
                    match = [(x, y)]
                last = puzzle_state[x][y]
            if len(match) > 2:
                matches.append(match)

        # Get all horizontal matches
        for y in range(height):
            match = []
            last = puzzle_state[0][y]
            match.append((0, y))
            for x in range(1, width):
                if puzzle_state[x][y] == last:
                    match.append((x, y))
                else:
                    if len(match) > 2:
                        matches.append(match)
                    match = [(x, y)]
                last = puzzle_state[x][y]
            if len(match) > 2:
                matches.append(match)

        return matches

    @staticmethod
    def _get_puzzle_state_after_move(
        puzzle_state: list[list[BackendGem]], move_action: MoveAction
    ) -> list[list[BackendGem]]:
        """
        Creates another view of the puzzle state after a specified move is
        applied.
        """
        # I just need to copy the lists inside the lists, not actually the
        # BackendGem objects themselves. But this solves the problem for now.
        # Can probably optimise later if more performance is needed.
        state_after_move = deepcopy(puzzle_state)
        apply_move_to_grid(state_after_move, move_action)
        return state_after_move

    def apply_explode_and_replace_phase(
        self, explode_and_replace_phase: ExplodeAndReplacePhase
    ) -> None:
        explode_gems = set(
            gem for match in explode_and_replace_phase.matches for gem in match
        )
        replacements = {
            col: replacements
            for col, replacements in explode_and_replace_phase.replacements
        }
        for x in range(self.width):
            for y in reversed(range(self.height)):
                if (x, y) in explode_gems:
                    self.puzzle_state[x].pop(y)
            if x in replacements:
                for replacement in replacements[x]:
                    self.puzzle_state[x].insert(0, BackendGem(gem_type=replacement))

    def reset(self) -> None:
        """
        If this function is called, a new bunch of backend gems is generated.
        """
        self.puzzle_state = self.get_initial_puzzle_state_with_no_matches(
            self.width, self.height
        )
