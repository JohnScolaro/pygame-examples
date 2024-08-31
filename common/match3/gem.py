import random
from enum import Enum, auto

import pygame
from pygame import Surface, Vector2

from common.assets.sounds.rock_clacks import NUM_ROCK_CLACKS_SOUNDS
from common.managers.managers import audio_manager
from common.match3.gem_assets import GemAssets
from common.match3.gem_types import GemTypes


class FrontendGem:

    ANIMATION_FPS = 30

    class FrontendGemState(Enum):
        IDLE = auto()
        FALLING = auto()
        EXPLODING = auto()

    def __init__(
        self,
        position: Vector2,
        gem_assets: GemAssets,
        gem_type: GemTypes,
        grid_width: int,
        grid_height: int,
        gem_width: int,
        gem_height: int,
    ) -> None:
        self.position = position
        self.gem_type = gem_type
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.gem_width = gem_width
        self.gem_height = gem_height
        self.state = self.FrontendGemState.IDLE

        # To handle movement when falling
        self.acceleration = pygame.Vector2(0, 1000)
        self.velocity = pygame.Vector2(0, 0)

        self.gem_frames = gem_assets.get_gem_frames(self.gem_type)
        self.frame_index = 0
        self.exploding_time = 0.0
        self.explosion_finished = False

    def process(self, delta: float) -> None:
        if self.state == self.FrontendGemState.IDLE:
            pass
        elif self.state == self.FrontendGemState.FALLING:
            self.process_falling(delta)
        elif self.state == self.FrontendGemState.EXPLODING:
            self.process_exploding(delta)

    def render(self, screen: Surface) -> None:
        screen.blit(self.gem_frames[self.frame_index], self.position)

        # Only draw fake wrap around gems when in an idle state. We don't want
        # wrap exploding gems, and when gems are falling, they're either being
        # spawned in from offscreen, or they're falling from ON the screen and
        # will never wrap.
        if self.state != self.FrontendGemState.IDLE:
            return

        if self.position.x < 0:
            screen.blit(
                self.gem_frames[self.frame_index],
                self.position + Vector2(self.grid_width * self.gem_width, 0),
            )
        if self.position.x > self.gem_width * (self.grid_width - 1):
            screen.blit(
                self.gem_frames[self.frame_index],
                self.position - Vector2(self.grid_width * self.gem_width, 0),
            )
        if self.position.y < 0:
            screen.blit(
                self.gem_frames[self.frame_index],
                self.position + Vector2(0, self.grid_height * self.gem_height),
            )
        if self.position.y > self.gem_height * (self.grid_height - 1):
            screen.blit(
                self.gem_frames[self.frame_index],
                self.position - Vector2(0, self.grid_height * self.gem_height),
            )

    @property
    def center(self) -> Vector2:
        center_x = self.position.x + self.gem_width / 2
        center_y = self.position.y + self.gem_height / 2
        return Vector2(center_x, center_y)

    def process_falling(self, delta: float) -> None:
        self.velocity += self.acceleration * delta
        self.position += self.velocity * delta

    def stop_falling(self) -> None:
        self.velocity = pygame.Vector2(0, 0)
        self.state = self.FrontendGemState.IDLE
        audio_manager.play_sound(
            f"rocks_clacking_{random.randint(0, NUM_ROCK_CLACKS_SOUNDS-1):02d}"
        )

    def set_exploding(self) -> None:
        self.state = self.FrontendGemState.EXPLODING
        self.exploding_time = 0.0

    def process_exploding(self, delta: float) -> None:
        self.exploding_time += delta
        frame = int(self.exploding_time / (1 / self.ANIMATION_FPS))
        if frame > len(self.gem_frames):
            self.explosion_finished = True
        self.frame_index = min(frame, len(self.gem_frames) - 1)

    def is_explosion_finished(self) -> bool:
        return self.explosion_finished
