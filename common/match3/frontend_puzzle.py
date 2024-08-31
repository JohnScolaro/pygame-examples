import dataclasses
import math
import random
from enum import Enum, auto
from itertools import pairwise
from typing import Literal

import pygame
import pytweening
from pygame import Surface, Vector2

from common.assets.sounds.combo_sounds import COMBO_SOUNDS
from common.assets.sounds.glass_clinks import GLASS_CLINKS, NUM_GLASS_CLINKS
from common.assets.sounds.rock_clacks import ROCK_CLACKS
from common.helpers import scale_and_clamp, wrapf
from common.managers.managers import audio_manager
from common.match3.backend_puzzle import BackendPuzzleState, Match
from common.match3.events import ScoreEvent, post_scoring_event
from common.match3.frontend_puzzle_particle_system import FrontendPuzzleParticleSystem
from common.match3.gem import FrontendGem
from common.match3.gem_assets import GemAssets
from common.match3.helpers import apply_move_to_grid
from common.match3.player_actions import MoveAction


class FrontendPuzzle:

    MAX_TWEEN_TIME = 0.6
    FALL_TIME = 0.5
    MAX_COMBO_LEVEL = 16

    # How long transitions (grid empty -> grid full and vice versa) should take.
    TRANSITION_TIME = 1.0

    class GameGridPauseState(Enum):
        """Whether or not the game is paused."""

        ACTIVE = auto()
        PAUSED = auto()

    class GameGridState(Enum):
        """The state of the grid and whether its filling or not."""

        EMPTY = auto()
        TRANSITIONING_TO_FULL = auto()
        FULL = auto()
        TRANSITIONING_TO_EMPTY = auto()

    @dataclasses.dataclass
    class RowColTransform:
        axis: Literal["row"] | Literal["col"]
        index: int
        magnitude: float

    @dataclasses.dataclass
    class TweenedRowColTransform:
        axis: Literal["row"] | Literal["col"]
        index: int
        max_magnitude: float
        seconds_since_release: float
        tween_time: float

    class MouseState(Enum):
        MOUSE_UP = auto()
        MOUSE_DOWN = auto()

    def __init__(
        self,
        position: Vector2,
        grid_width: int,
        grid_height: int,
        initial_state: Literal["full", "empty"],
    ) -> None:
        self.position = position

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.gem_width = 30
        self.gem_height = 30

        self.gem_assets = GemAssets()

        self.pause_state = self.GameGridPauseState.ACTIVE
        self.grid_state = (
            self.GameGridState.EMPTY
            if initial_state == "empty"
            else self.GameGridState.FULL
        )
        self.transition_timer = 0.0
        self.mouse_state = self.MouseState.MOUSE_UP

        self.gem_grid_surface = Surface(self.get_surface_size(), flags=pygame.SRCALPHA)

        self.grab_distance = pygame.Vector2(0, 0)
        self.clicked = False
        self.click_initial_position = Vector2(0, 0)
        self.clicked_gem = (0, 0)

        self.accepting_inputs = True

        self.backend_puzzle_state = BackendPuzzleState(
            self.grid_width, self.grid_height
        )

        # Combo logic
        self.combo_level = 0
        self.combo_active = False

        # Create the frontend puzzle state from the backend puzzle state.
        self.gems: list[list[FrontendGem]] = self.get_frontend_gems()

        self.snapping_row_col_transforms: list[
            FrontendPuzzle.TweenedRowColTransform
        ] = []
        self.grabbed_row_col_transform: FrontendPuzzle.RowColTransform | None = None

        self.applied_actions_waiting_for_explosions_to_complete: list[MoveAction] = []
        self.explosions: list[FrontendGem] = []

        # Particle System
        self.particle_system = FrontendPuzzleParticleSystem(
            self.position,
            gem_width=self.gem_width,
            gem_height=self.gem_height,
            border=10,
        )

        # Load required audio
        for i, path in enumerate(GLASS_CLINKS):
            audio_manager.load_sound(f"glass_clink_{i}", path)
        for i, path in enumerate(ROCK_CLACKS):
            audio_manager.load_sound(f"rocks_clacking_{i:02d}", path)
        for i, path in enumerate(COMBO_SOUNDS):
            audio_manager.load_sound(f"combo_sound_{i:02d}", path)
            audio_manager.set_sound_volume(f"combo_sound_{i:02d}", 0.2)

    def get_surface_size(self) -> tuple[int, int]:
        return (self.grid_width * self.gem_width, self.grid_height * self.gem_height)

    def get_gem_default_position_from_grid_coord(
        self, grid_coord: tuple[int, int]
    ) -> Vector2:
        return Vector2(self.gem_width * grid_coord[0], self.gem_height * grid_coord[1])

    def process(self, delta: float) -> None:
        if self.pause_state == FrontendPuzzle.GameGridPauseState.PAUSED:
            return

        if self.grid_state == FrontendPuzzle.GameGridState.TRANSITIONING_TO_EMPTY:
            self.process_transition_to_empty_animation(delta)

        for column in self.gems:
            for gem in column:
                gem.process(delta)

        self.process_exploding_gems(delta)
        self.process_snapping_row_col_tweens(delta)

        if self.is_ready_for_next_explode_and_replace_phase():
            self.frontend_handle_explode_and_replace_phase()

        gem_positions = self.get_position_of_gems()

        # Set positions of gems to the gem_positions
        for gem_col, position_col in zip(self.gems, gem_positions):
            for gem, position in zip(gem_col, position_col):
                if gem.state == FrontendGem.FrontendGemState.IDLE:
                    gem.position = position
                elif gem.state == FrontendGem.FrontendGemState.FALLING:
                    if (
                        self.grid_state
                        != FrontendPuzzle.GameGridState.TRANSITIONING_TO_EMPTY
                    ):
                        if gem.position.y >= position.y:
                            gem.stop_falling()

        # If we are transitioning to full, when all the gems are idle, we're full :)
        if self.grid_state == FrontendPuzzle.GameGridState.TRANSITIONING_TO_FULL:
            if all(
                gem.state == FrontendGem.FrontendGemState.IDLE
                for gem_col in self.gems
                for gem in gem_col
            ):
                self.grid_state = FrontendPuzzle.GameGridState.FULL

        self.particle_system.process(delta)

    def render(self, screen: Surface) -> None:
        # If the grid is empty, just don't render. There shouldn't be anything
        # to render anyway, and we run into problems trying to render the match
        # lines for frontend gems that don't exist.
        if self.grid_state == self.GameGridState.EMPTY:
            return

        # Blank the gem grid surface, ready for drawing
        self.gem_grid_surface.fill((0, 0, 0, 0))

        self.particle_system.render(screen)
        self.draw_match_lines()
        self.draw_frontend_gems()

        # Draw the gem grid surface to the screen
        screen.blit(self.gem_grid_surface, self.position)

    def draw_match_lines(self) -> None:
        row_col_transform = self.get_row_col_transform_from_drag_vector(
            self.grab_distance
        )
        move_action = self.get_move_from_row_col_transform(row_col_transform)
        matches = self.backend_puzzle_state.get_matches_from_hypothetical_move(
            move_action
        )

        for match in matches:
            if not self.should_draw_match_lines_for_this_match(
                match, row_col_transform
            ):
                continue

            actual_match = self.get_actual_match_of_gems_in_hypothetical_match(
                match, move_action, self.grid_width, self.grid_height
            )
            for coord_1, coord_2 in pairwise(actual_match):
                fraction = self.get_linear_fraction_to_closest_snap(
                    row_col_transform, self.gem_width, self.gem_height
                )
                line_width = self.get_match_line_thickness_from_fraction(fraction)
                pygame.draw.line(
                    self.gem_grid_surface,
                    "white",
                    self.gems[coord_1[0]][coord_1[1]].center,
                    self.gems[coord_2[0]][coord_2[1]].center,
                    width=line_width,
                )

    @staticmethod
    def should_draw_match_lines_for_this_match(
        match: Match, row_col_transform: RowColTransform
    ) -> bool:
        """
        Given a match and the RowColTransform of the currently grabbed row/col,
        return a boolean. If True, we should draw the white lines under the
        gems in the match. If False, we should not.

        This is implemented such that lines are only drawn on matches that are
        caused as a direct result of manually moving a row/col.
        """
        for gem in match:
            if row_col_transform.axis == "row":
                index = 1
            else:
                index = 0
            if (
                row_col_transform.index == gem[index]
                and row_col_transform.magnitude != 0
            ):
                return True
        return False

    def draw_frontend_gems(self) -> None:
        # Draw exploding gems
        for gem in self.explosions:
            gem.render(self.gem_grid_surface)

        # Draw the gems in the grid.
        for row in self.gems:
            for gem in row:
                gem.render(self.gem_grid_surface)

    @staticmethod
    def get_actual_match_of_gems_in_hypothetical_match(
        match: Match, move_action: MoveAction, grid_width: int, grid_height: int
    ) -> list[tuple[int, int]]:
        """
        In the frontend, we have the state of our gems, and we know what
        matches we can get given a hypothetical move, but then we need to get
        the index of the ACTUAL frontend gems from the hypothetical match
        index. This is the function that does that.
        """
        modified_match: list[tuple[int, int]] = []
        for coords in match:
            if move_action.row_or_col == "row":
                if coords[1] == move_action.index:
                    modified_match.append(
                        ((coords[0] - move_action.amount) % grid_width, coords[1])
                    )
                else:
                    modified_match.append(coords)
            else:
                if coords[0] == move_action.index:
                    modified_match.append(
                        (coords[0], (coords[1] + move_action.amount) % grid_height)
                    )
                else:
                    modified_match.append(coords)

        return modified_match

    def get_position_of_gems(self) -> list[list[pygame.Vector2]]:
        # First, get all the default positions
        gem_positions = []
        for x in range(self.grid_width):
            row = []
            for y in range(self.grid_height):
                row.append(self.get_gem_default_position_from_grid_coord((x, y)))
            gem_positions.append(row)

        gem_positions = self.apply_currently_snapping_row_col_transforms(gem_positions)
        gem_positions = self.apply_currently_grabbed_row_col_transform(gem_positions)
        return gem_positions

    def apply_currently_snapping_row_col_transforms(
        self, gem_positions: list[list[pygame.Vector2]]
    ) -> list[list[pygame.Vector2]]:
        for tweened_row_col_transform in self.snapping_row_col_transforms:
            if tweened_row_col_transform.tween_time == 0:
                tween_multiplier = 1
            else:
                tween_multiplier = pytweening.easeInCubic(
                    tweened_row_col_transform.seconds_since_release
                    / tweened_row_col_transform.tween_time
                )

            distance = tweened_row_col_transform.max_magnitude * (1 - tween_multiplier)
            gem_positions = self.apply_row_col_transform_to_gem_positions(
                gem_positions,
                self.RowColTransform(
                    axis=tweened_row_col_transform.axis,
                    index=tweened_row_col_transform.index,
                    magnitude=distance,
                ),
            )
        return gem_positions

    def apply_currently_grabbed_row_col_transform(
        self, gem_positions: list[list[pygame.Vector2]]
    ) -> list[list[pygame.Vector2]]:
        # If it exists, apply the currently grabbed row_col_transform (assuming it's valid)
        if self.grab_distance:
            row_col_transform = self.get_row_col_transform_from_drag_vector(
                self.grab_distance
            )
            if self.is_valid_row_col_transform_given_snapping_transforms(
                self.snapping_row_col_transforms, row_col_transform
            ):
                gem_positions = self.apply_row_col_transform_to_gem_positions(
                    gem_positions, row_col_transform
                )
        return gem_positions

    def process_exploding_gems(self, delta: float) -> None:
        # For now, exploding gems just disappear. Can improve animations later.
        for i in reversed(range(len(self.explosions))):
            exploding_gem = self.explosions[i]
            exploding_gem.process_exploding(delta)
            if exploding_gem.is_explosion_finished():
                self.explosions.pop(i)

    def process_snapping_row_col_tweens(self, delta: float) -> None:
        completed_tween_indexes = []

        for i, tweened_row_col_transform in enumerate(self.snapping_row_col_transforms):
            tweened_row_col_transform.seconds_since_release += delta

            if (
                tweened_row_col_transform.seconds_since_release
                >= tweened_row_col_transform.tween_time
            ):
                completed_tween_indexes.append(i)
                random_clink = random.randint(0, NUM_GLASS_CLINKS - 1)
                clink_sound = f"glass_clink_{random_clink}"
                audio_manager.set_sound_volume(
                    clink_sound,
                    min(
                        1.0,
                        abs(tweened_row_col_transform.max_magnitude)
                        / (self.gem_height * 3),
                    ),
                )
                audio_manager.play_sound(clink_sound)

        for i in reversed(completed_tween_indexes):
            self.snapping_row_col_transforms.pop(i)

    def handle_event(self, event) -> bool:
        # If the grid is empty, filling, or emptying, we don't want to process
        # any events for it, and just let it animate.
        if self.grid_state != self.GameGridState.FULL:
            return False

        # Also if we're not accepting inputs, just do nothing. Sometimes when
        # the grid is full, we also don't want to accept inputs. For example,
        # when the grid is full, and the countdown is counting, but the round
        # hasn't started yet.
        if not self.accepting_inputs:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            clicked_in_gem_grid = (
                self.gem_grid_surface.get_rect()
                .move(self.position.x, self.position.y)
                .collidepoint(event.pos)
            )
            if clicked_in_gem_grid:
                self.click_initial_position = Vector2(event.pos)
                self.set_clicked_gem(
                    self.get_clicked_gem_from_mouse_position(event.pos)
                )
                self.mouse_state = self.MouseState.MOUSE_DOWN
                return True
            else:
                return False

        if event.type == pygame.MOUSEBUTTONUP:
            if self.mouse_state == self.MouseState.MOUSE_DOWN:
                self.mouse_state = self.MouseState.MOUSE_UP

                # Get the current move/drag distance and put it into the queue of snapping row/cols
                row_col_transform = self.get_row_col_transform_from_drag_vector(
                    self.grab_distance
                )
                # Get the move action from the current transform.
                move_action = self.get_move_from_row_col_transform(row_col_transform)
                # Get any hypothetical matches from that move action
                matches = self.backend_puzzle_state.get_matches_from_hypothetical_move(
                    move_action
                )

                # If there are matches
                if matches:
                    # Add the move to a list of moves to apply to the backend
                    # all at once when animations have finished
                    self.applied_actions_waiting_for_explosions_to_complete.append(
                        move_action
                    )

                    # Apply the move to the frontend
                    self.frontend_apply_move_actions([move_action])

                    # Add a tweened row col transform to move the row/col from
                    # where it was dropped, to it's new match location.
                    self.snapping_row_col_transforms.append(
                        self.get_magnitude_to_snap_to_currently_hovered_position(
                            row_col_transform
                        )
                    )

                # If there aren't, snap it back to where it was before.
                else:
                    if self.is_valid_row_col_transform_given_snapping_transforms(
                        self.snapping_row_col_transforms, row_col_transform
                    ):
                        self.snapping_row_col_transforms.append(
                            self.TweenedRowColTransform(
                                axis=row_col_transform.axis,
                                index=row_col_transform.index,
                                max_magnitude=row_col_transform.magnitude,
                                seconds_since_release=0.0,
                                tween_time=scale_and_clamp(
                                    self.gem_height * 3,
                                    0.0,
                                    self.MAX_TWEEN_TIME,
                                    self.MAX_TWEEN_TIME / 4,
                                    abs(row_col_transform.magnitude),
                                ),
                            )
                        )

                self.grab_distance = pygame.Vector2(0, 0)
                return True
            else:
                return False

        if event.type == pygame.MOUSEMOTION:
            if self.mouse_state == self.MouseState.MOUSE_DOWN:
                pos = Vector2(event.pos)
                self.grab_distance = pos - self.click_initial_position
                self.grabbed_row_col_transform = (
                    self.get_row_col_transform_from_drag_vector(self.grab_distance)
                )
                return True
            else:
                return False

        return False

    def set_clicked_gem(self, clicked_gem: tuple[int, int]) -> None:
        self.clicked_gem = clicked_gem

    def get_clicked_gem_from_mouse_position(
        self, mouse_pos: tuple[int, int]
    ) -> tuple[int, int]:
        relative_pos = mouse_pos - self.position
        return (
            int(relative_pos[0] // self.gem_width),
            int(relative_pos[1] // self.gem_height),
        )

    def get_row_col_transform_from_drag_vector(
        self, drag_vector: pygame.Vector2
    ) -> RowColTransform:
        angle_in_rads = math.radians(Vector2(1, 0).angle_to(drag_vector))
        directional_scaling_factor = abs(math.cos(angle_in_rads * 2))

        if abs(drag_vector.x) > abs(drag_vector.y):
            return self.RowColTransform(
                axis="row",
                index=self.clicked_gem[1],
                magnitude=drag_vector.x * directional_scaling_factor,
            )
        else:
            return self.RowColTransform(
                axis="col",
                index=self.clicked_gem[0],
                magnitude=drag_vector.y * directional_scaling_factor,
            )

    def apply_row_col_transform_to_gem_positions(
        self,
        gem_positions: list[list[pygame.Vector2]],
        row_col_transform: RowColTransform,
    ) -> list[list[pygame.Vector2]]:
        if row_col_transform.axis == "row":
            gem_positions = self.apply_row_transform(
                gem_positions, row_col_transform.index, row_col_transform.magnitude
            )
        else:
            gem_positions = self.apply_col_transform(
                gem_positions, row_col_transform.index, row_col_transform.magnitude
            )
        return gem_positions

    def apply_col_transform(
        self,
        gem_positions: list[list[pygame.Vector2]],
        col_index: int,
        magnitude: float,
    ) -> list[list[pygame.Vector2]]:
        for y in range(self.grid_height):
            gem_position = gem_positions[col_index][y]
            new_y_position = gem_position.y + magnitude
            wrapped_new_y_position = wrapf(
                new_y_position,
                self.get_gem_default_position_from_grid_coord((col_index, -1)).y
                + self.gem_height / 2,
                self.get_gem_default_position_from_grid_coord(
                    (col_index, self.grid_height - 1)
                ).y
                + self.gem_height / 2,
            )
            gem_position.y = wrapped_new_y_position
        return gem_positions

    def apply_row_transform(
        self,
        gem_positions: list[list[pygame.Vector2]],
        row_index: int,
        magnitude: float,
    ) -> list[list[pygame.Vector2]]:
        for x in range(self.grid_width):
            gem_position = gem_positions[x][row_index]
            new_x_position = gem_position.x + magnitude
            wrapped_new_x_position = wrapf(
                new_x_position,
                self.get_gem_default_position_from_grid_coord((0, row_index)).x
                - self.gem_width / 2,
                self.get_gem_default_position_from_grid_coord(
                    (self.grid_width - 1, row_index)
                ).x
                + self.gem_width / 2,
            )
            gem_position.x = wrapped_new_x_position
        return gem_positions

    def get_move_from_row_col_transform(
        self, row_col_transform: RowColTransform
    ) -> MoveAction:
        row_or_col = row_col_transform.axis
        if row_or_col == "row":
            amount = (
                (row_col_transform.magnitude - (self.gem_width / 2)) // self.gem_width
            ) + 1
        else:
            amount = (
                (row_col_transform.magnitude - (self.gem_height / 2)) // self.gem_height
            ) + 1
            amount *= -1

        return MoveAction(
            row_or_col=row_or_col, index=row_col_transform.index, amount=int(amount)
        )

    @staticmethod
    def get_linear_fraction_to_closest_snap(
        row_col_transform: RowColTransform, gem_width: int, gem_height: int
    ) -> float:
        """
        When you're dragging a col/row, sometimes the col/row you're dragging
        aligns perfectly with the other rows and cols, at other times, it's
        exactly in the middle between rows. This function returns 1 when it's
        exactly in the middle, and 0 when it's perfectly aligned so you can
        animate things based on that.
        """
        if row_col_transform.axis == "row":
            amount = row_col_transform.magnitude % gem_width
            return abs(1 - (abs((gem_width / 2) - amount) / (gem_width / 2)))
        else:
            amount = row_col_transform.magnitude % gem_height
            return abs(1 - (abs((gem_height / 2) - amount) / (gem_height / 2)))

    @staticmethod
    def get_match_line_thickness_from_fraction(fraction: float) -> int:
        """
        Where fraction is a float from 0 (meaning gems are perfectly aligned),
        to 1, meaning gems are not aligned at all, get the line thickness of
        the "match line" to draw between the gems.
        """
        return int(5 * (1 - fraction))

    @staticmethod
    def is_valid_row_col_transform_given_snapping_transforms(
        snapping_row_col_transforms: list[TweenedRowColTransform],
        row_col_transform: RowColTransform,
    ) -> bool:
        """
        You shouldn't be able to drag rows when cols are already snapping,
        because then you'll have one gem out of alignment. Vice-versa for cols
        when rows are snapping too.
        """
        # Because of this rule, all row/col snapping transforms are going to
        # be in the same axis, so we only need to check the first one.
        if snapping_row_col_transforms:
            if snapping_row_col_transforms[0].axis != row_col_transform.axis:
                return False

        # You also can not have multiple row/col transforms to the same row/col
        for snapping_row_col_transform in snapping_row_col_transforms:
            if snapping_row_col_transform.index == row_col_transform.index:
                return False

        return True

    def frontend_apply_move_actions(self, move_actions: list[MoveAction]) -> None:
        """
        Apply the supplied move actions to the frontend gem grid.
        """
        for move_action in move_actions:
            apply_move_to_grid(self.gems, move_action)

    def frontend_handle_explode_and_replace_phase(self) -> None:
        """
        When we submit moves to the backend puzzle and get back an
        ExplodeAndReplacePhase, this kicks off a few things on the frontend
        side.
        """
        explode_and_replace_phase = (
            self.backend_puzzle_state.get_next_explode_and_replace_phase(
                self.applied_actions_waiting_for_explosions_to_complete
            )
        )
        self.applied_actions_waiting_for_explosions_to_complete = []

        if explode_and_replace_phase.is_nothing_to_do():
            self.combo_level = 0
            self.combo_active = False
            return

        # Create and post scoring event
        post_scoring_event(
            ScoreEvent(
                scoring_matches=[
                    (match, self.gems[match[0][0]][match[0][1]].gem_type)
                    for match in explode_and_replace_phase.matches
                ],
                combo_level=self.combo_level,
            )
        )

        # Remove gems from gem grid and insert into the list of exploding gems.
        gems_to_explode = set(
            gem for match in explode_and_replace_phase.matches for gem in match
        )

        self.set_all_gems_above_exploding_gems_to_falling_state(gems_to_explode)

        # Move the gems that are exploding from the gem grid to the explosions list.
        for gem in reversed(sorted(gems_to_explode, key=lambda x: (x[1], x[0]))):
            # and put it in the explosion state
            exploding_gem = self.gems[gem[0]].pop(gem[1])
            exploding_gem.set_exploding()
            self.explosions.append(exploding_gem)
            self.particle_system.create_explosion(
                self.get_gem_default_position_from_grid_coord(gem),
                5,
                exploding_gem.gem_type,
            )

        # Handle combos
        if self.combo_active:
            self.combo_level += 1
        else:
            self.combo_active = True
        audio_manager.play_sound(
            f"combo_sound_{min(self.combo_level, self.MAX_COMBO_LEVEL - 1):02d}"
        )

        # Put replacement gems into the list of falling replacement gems.
        replacement_gems = {
            col: replacement_gem_types
            for col, replacement_gem_types in explode_and_replace_phase.replacements
        }

        # Put the replacement gems into the gem grid, and set their positions
        # to above the grid and their states to falling.
        for col, replacement_gem_types in replacement_gems.items():
            for i, gem_type in enumerate(replacement_gem_types):
                replacement_frontend_gem = FrontendGem(
                    position=self.get_gem_default_position_from_grid_coord(
                        (col, -i - 1)
                    ),
                    gem_assets=self.gem_assets,
                    gem_type=gem_type,
                    grid_width=self.grid_width,
                    grid_height=self.grid_height,
                    gem_width=self.gem_width,
                    gem_height=self.gem_height,
                )
                replacement_frontend_gem.state = FrontendGem.FrontendGemState.FALLING
                self.gems[col].insert(0, replacement_frontend_gem)

    def set_all_gems_above_exploding_gems_to_falling_state(
        self, gems_to_explode: set[tuple[int, int]]
    ) -> None:
        lowest_exploding_gem_from_each_column: dict[int, int] = {}
        for x, y in gems_to_explode:
            lowest_exploding_gem_from_each_column[x] = max(
                lowest_exploding_gem_from_each_column.get(x, -1), y
            )

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if x in lowest_exploding_gem_from_each_column:
                    if y < lowest_exploding_gem_from_each_column[x]:
                        self.gems[x][y].state = FrontendGem.FrontendGemState.FALLING

    def is_ready_for_next_explode_and_replace_phase(self) -> bool:
        """
        Return True if the frontend is ready to start animating the next
        explode and replace phase.
        """
        full_state = self.grid_state == FrontendPuzzle.GameGridState.FULL

        nothing_falling = not any(
            gem.state == FrontendGem.FrontendGemState.FALLING
            for col in self.gems
            for gem in col
        )

        return full_state and nothing_falling

    def get_magnitude_to_snap_to_currently_hovered_position(
        self, row_col_transform: RowColTransform
    ) -> TweenedRowColTransform:
        """
        When you drag a row/col it creates a RowColTransform. If you drag it to
        a position that results in a match, you want to snap it to that match.
        This function calculates the TweenedRowColTransform that snaps from the
        position you un-clicked, back to the grid. Maybe this illustration will
        show what I mean?

                    ┌──────┬──────┬──────┬──────┬──────┐
                    │  g1  │  g2  │  g3  │  g4  │  g5  │
                                  ◄───────────────
                                ─►   Original Drag
                          Generated Tween
        """
        if row_col_transform.axis == "row":
            magnitude = self.gem_width / 2 - (
                row_col_transform.magnitude % self.gem_width
            )

        else:
            magnitude = self.gem_height / 2 - (
                row_col_transform.magnitude % self.gem_height
            )

        if magnitude < 0:
            magnitude = -15 - magnitude
        else:
            magnitude = 15 - magnitude

        return self.TweenedRowColTransform(
            axis=row_col_transform.axis,
            index=row_col_transform.index,
            max_magnitude=magnitude,
            seconds_since_release=0.0,
            tween_time=scale_and_clamp(
                self.gem_height * 3,
                0.0,
                self.MAX_TWEEN_TIME,
                self.MAX_TWEEN_TIME / 4,
                abs(row_col_transform.magnitude),
            ),
        )

    def pause_frontend_puzzle(self) -> None:
        self.pause_state = FrontendPuzzle.GameGridPauseState.PAUSED

    def resume_frontend_puzzle(self) -> None:
        self.pause_state = FrontendPuzzle.GameGridPauseState.ACTIVE

    def get_frontend_puzzle_state(self) -> GameGridPauseState:
        return self.pause_state

    def toggle_frontend_puzzle_state(self) -> None:
        if self.pause_state == FrontendPuzzle.GameGridPauseState.PAUSED:
            self.pause_state = FrontendPuzzle.GameGridPauseState.ACTIVE
        else:
            self.pause_state = FrontendPuzzle.GameGridPauseState.PAUSED

    def fill_grid(self) -> None:
        if self.grid_state != FrontendPuzzle.GameGridState.EMPTY:
            raise Exception(
                "You should only fill the grid when its empty. We shouldn't ever get here."
            )

        # Reset the backend gem state so we get a fresh new frontend grid
        # every time we fill the grid.
        self.backend_puzzle_state.reset()
        self.grid_state = self.GameGridState.TRANSITIONING_TO_FULL
        self.transition_timer = 0.0
        self.gems = self.get_frontend_gems()

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                gem_num = x * self.grid_width - y + self.grid_height
                self.gems[x][y].position.y -= self.gem_height * self.grid_height
                self.gems[x][y].state = FrontendGem.FrontendGemState.FALLING
                self.gems[x][y].velocity = pygame.Vector2(0, -gem_num * 10)

    def get_frontend_gems(self) -> list[list[FrontendGem]]:
        """
        Create a 2D array of frontend gems and return it.
        """
        gems = []
        for x in range(self.grid_width):
            row = []
            for y in range(self.grid_height):
                frontend_gem = FrontendGem(
                    position=pygame.Vector2(
                        x * self.gem_width,
                        y * self.gem_height,
                    ),
                    gem_assets=self.gem_assets,
                    gem_type=self.backend_puzzle_state.puzzle_state[x][y].gem_type,
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                    gem_height=self.gem_height,
                    gem_width=self.gem_width,
                )
                row.append(frontend_gem)
            gems.append(row)
        return gems

    def empty_grid(self) -> None:
        if self.grid_state != FrontendPuzzle.GameGridState.FULL:
            raise Exception(
                "You should only empty the grid when its full. We shouldn't ever get here."
            )

        self.grid_state = self.GameGridState.TRANSITIONING_TO_EMPTY
        self.transition_timer = 0.0

    def process_transition_to_empty_animation(self, delta: float) -> None:
        # Increment the timer
        self.transition_timer += delta

        # Set rows to falling one at a time
        rows_to_set_to_falling = int(self.transition_timer / 0.1)
        for x in range(min(self.grid_width, rows_to_set_to_falling)):
            for y in range(self.grid_height):
                self.gems[x][y].state = FrontendGem.FrontendGemState.FALLING

        # If all the rows are below the screen, set the state to empty.
        if all(
            gem.position.y >= ((self.grid_height + 1) * self.gem_height)
            for gem_col in self.gems
            for gem in gem_col
        ):
            self.gems = []
            self.grid_state = FrontendPuzzle.GameGridState.EMPTY

    def set_accepting_inputs(self, is_accepting_inputs: bool) -> None:
        self.accepting_inputs = is_accepting_inputs
