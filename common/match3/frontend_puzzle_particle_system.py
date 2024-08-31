import math
import random

import pygame

from common.match3.gem_types import GemTypes


class FrontendPuzzleParticleSystem:
    def __init__(
        self,
        position: pygame.Vector2,
        gem_width: int,
        gem_height: int,
        border: int,
    ) -> None:
        self.position = position
        self.gem_width = gem_width
        self.gem_height = gem_height

        self.particles: list[ImmediateBlastParticle] = []

    def process(self, delta: float) -> None:
        for particle in self.particles:
            particle.process(delta)
        for i in reversed(range(len(self.particles))):
            particle = self.particles[i]
            if particle.is_dead():
                self.particles.pop(i)

    def render(self, screen: pygame.Surface) -> None:
        for particle in self.particles:
            particle.draw(screen)

    def create_explosion(
        self, explosion_location: pygame.Vector2, num_particles: int, gem_type: GemTypes
    ) -> None:
        for i in range(num_particles):
            self.particles.append(
                ImmediateBlastParticle(
                    explosion_location
                    + self.position
                    + pygame.Vector2(self.gem_width / 2, self.gem_height / 2),
                    4,
                    GemTypes.get_color_from_gem_type(gem_type),
                    lifespan=0.5,
                )
            )


class ImmediateBlastParticle:
    def __init__(
        self,
        initial_position: pygame.Vector2,
        max_radius: int,
        color: pygame.Color,
        lifespan: float,
    ):
        self.position = initial_position
        self.max_radius = max_radius
        self.color = color
        self.lifespan = 0.3
        self.age = 0.0

        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * 150

    def process(self, delta):
        self.position += self.velocity * delta
        self.age += delta

    def draw(self, screen):
        radius = self.max_radius * (1 - (self.age / self.lifespan))
        pygame.draw.circle(
            screen,
            pygame.Color("white"),
            (self.position.x, self.position.y),
            radius,
        )

    def is_dead(self) -> bool:
        return self.age > self.lifespan
