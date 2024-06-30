import os
import random
from enum import Enum, auto

import cv2
import numpy as np
import pygame
import pytweening

from common.assets.fonts import M5X7
from common.constants import SCREEN_HEIGHT, SCREEN_WIDTH
from common.file_helpers import get_cards_directory, get_sounds_directory
from common.helpers import MovingAverage, Tweener
from common.managers.managers import audio_manager, font_manager


class Card:

    TIME_TO_SNAP_TO_CURSOR_ON_PICKUP = 0.2  # Seconds
    MAX_CARD_ROTATION = 50  # Degrees

    class CardState(Enum):
        IDLE = auto()
        GRABBED = auto()

    def __init__(
        self,
        position: pygame.Vector2,
        size: pygame.Vector2,
        background_surface: pygame.Surface,
    ) -> None:
        self.position = position
        self.size = size
        self.background_surface = background_surface

        self.state = self.CardState.IDLE
        self.card_goal_position: pygame.Vector2 = position
        self.card_pickup_position: pygame.Vector2 = pygame.Vector2(0, 0)
        self.tweener_snap_to_mouse = Tweener(
            1.0, 0.0, self.TIME_TO_SNAP_TO_CURSOR_ON_PICKUP, pytweening.easeInCubic
        )
        self.average_velocity_x = MovingAverage(size=10)
        self.average_velocity_y = MovingAverage(size=10)
        self.velocity: pygame.Vector2 = pygame.Vector2(0, 0)

        self.points = np.array(
            [
                [-self.size.x / 2, self.size.y / 2, 0],
                [self.size.x / 2, self.size.y / 2, 0],
                [self.size.x / 2, -self.size.y / 2, 0],
                [-self.size.x / 2, -self.size.y / 2, 0],
            ]
        )
        self.card_rotation = pygame.Vector2(0, 0)

        self.surface = pygame.Surface(self.size)

    def render(self, screen: pygame.Surface, delta: float) -> None:
        self.update_position_and_velocity(delta)
        self.update_card_rotation(delta)

        self.surface.fill("purple")
        self.surface.blit(self.background_surface, (0, 0))

        points = self.get_screen_coord_points()

        self.blit_perspective_transformed_card_surface(screen, points)
        self.blit_card_outline(screen, points)

    def update_position_and_velocity(self, delta: float) -> None:
        # Update tweener
        self.tweener_snap_to_mouse.update(delta)

        # Get new position
        if self.tweener_snap_to_mouse.is_finished():
            new_position = self.card_goal_position
        else:
            new_position = self.card_goal_position - (
                (self.tweener_snap_to_mouse.get_value())
                * (self.card_goal_position - self.card_pickup_position)
            )

        # Calculate velocity based on an average of the last few position
        # changes. I only do this because using instantaneous position is super
        # jerky because it's essentially zero except the frame/subframe when a
        # mouse event is received.
        self.average_velocity_x.add((new_position.x - self.position.x) / delta)
        self.average_velocity_y.add((new_position.y - self.position.y) / delta)
        x_velocity = self.average_velocity_x.average()
        y_velocity = self.average_velocity_y.average()

        # If the velocity is below a threshold, snap it to 0. This stops float
        # errors where the velocity just stays at 1e-13 or something instead of
        # returning to zero, and the card looks rotated at rest.
        VELOCITY_EPSILON = 1
        self.velocity = pygame.Vector2(
            x_velocity if abs(x_velocity) > VELOCITY_EPSILON else 0,
            y_velocity if abs(y_velocity) > VELOCITY_EPSILON else 0,
        )

        # Set position
        self.position = new_position

    def update_card_rotation(self, delta: float) -> None:
        card_rotation_vector = pygame.Vector2(
            -self.velocity.x / 15, self.velocity.y / 15
        )

        ROTATION_EPSILON = 4
        # Rotations below this magnitude are snapped to zero because tiny
        # rotations stop us aligning cards nicely when moving cards around
        # slowly.
        if card_rotation_vector.magnitude() < ROTATION_EPSILON:
            card_rotation_vector = pygame.Vector2(0, 0)
        elif card_rotation_vector.magnitude() > self.MAX_CARD_ROTATION:
            card_rotation_vector = card_rotation_vector / (
                card_rotation_vector.magnitude() / self.MAX_CARD_ROTATION
            )

        self.card_rotation_degrees_y = card_rotation_vector.x
        self.card_rotation_degrees_x = card_rotation_vector.y

    @staticmethod
    def get_bounding_box_dimensions(points: np.ndarray) -> tuple[int, int]:
        if points.size == 0:
            return 0, 0

        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        return int(width), int(height)

    def rotate_points(
        self, points: list[np.ndarray], degrees_x: float, degrees_y: float
    ) -> np.ndarray:
        # Convert angles from degrees to radians
        degrees_x = np.radians(degrees_x)
        degrees_y = np.radians(degrees_y)

        # Rotation matrix around the x-axis
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(degrees_x), -np.sin(degrees_x)],
                [0, np.sin(degrees_x), np.cos(degrees_x)],
            ]
        )

        # Rotation matrix around the y-axis
        Ry = np.array(
            [
                [np.cos(degrees_y), 0, np.sin(degrees_y)],
                [0, 1, 0],
                [-np.sin(degrees_y), 0, np.cos(degrees_y)],
            ]
        )

        # Apply the rotations
        rotated_points = []
        for point in points:
            rotated_point = np.dot(Rx, point)
            rotated_point = np.dot(Ry, rotated_point)
            rotated_points.append(rotated_point)

        return np.array(rotated_points)

    def perspective_projection(self, points: np.ndarray, d: float) -> np.ndarray:
        projected_points = []
        for point in points:
            x, y, z = point
            x_p = x * d / (z + d)
            y_p = y * d / (z + d)
            projected_points.append([x_p, y_p])

        return np.array(projected_points)

    def get_screen_coord_points(self) -> np.ndarray:
        points = []
        rotated_points = self.rotate_points(
            self.points, self.card_rotation_degrees_x, self.card_rotation_degrees_y
        )
        projected_points = self.perspective_projection(rotated_points, 150)
        for point in projected_points:
            points.append([point[0], point[1]])
        return np.array(points)

    def calculate_perspective_transform(
        self, src: np.ndarray, dst: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the perspective transform matrix from src to dst points.

        Parameters:
        src (numpy.ndarray): Source points, shape (4, 2).
        dst (numpy.ndarray): Destination points, shape (4, 2).

        Returns:
        numpy.ndarray: The perspective transformation matrix, shape (3, 3).
        """

        src_copy = src[:, :2].copy()
        src_copy += np.array([self.size.x / 2, self.size.y / 2])

        dst_copy = dst.copy()
        min_values = dst_copy.min(axis=0)

        dst_copy -= min_values

        return cv2.getPerspectiveTransform(
            src_copy.astype(np.float32), dst_copy.astype(np.float32)
        )

    def blit_perspective_transformed_card_surface(
        self, surface: pygame.Surface, points: np.ndarray
    ) -> None:
        # Calculate the transform matrix required to transform the surface.
        transform = self.calculate_perspective_transform(self.points, points)

        transformed_surface = self.apply_perspective_transform(
            self.surface,
            transform,
            self.get_bounding_box_dimensions(points),
        )

        # Transform the top left point so we know the end location of the top left point.
        top_left_point = np.array([0, 0])
        top_left = self.transform_point(top_left_point, transform)

        surface.blit(
            transformed_surface,
            self.position - top_left + points[3],
        )

    def blit_card_outline(self, surface: pygame.Surface, points: np.ndarray) -> None:
        # Iterate through pairs including the wrap-around pair
        modified_points = points.tolist()
        modified_points[0][1] -= 1
        modified_points[1][0] -= 1
        modified_points[1][1] -= 1
        modified_points[2][0] -= 1
        for i in range(len(modified_points)):
            a, b = (modified_points[i], modified_points[(i + 1) % len(modified_points)])

            pygame.draw.line(
                surface,
                pygame.Color(0, 0, 0, 100),
                a + self.position,
                b + self.position,
                1,
            )

    def apply_perspective_transform(
        self, surface: pygame.Surface, matrix: np.ndarray, dst_size: tuple[int, int]
    ) -> pygame.Surface:
        """
        Apply a perspective transform to a Pygame surface using a transformation matrix.

        Parameters:
        surface (pygame.Surface): The source Pygame surface.
        matrix (numpy.ndarray): The 3x3 transformation matrix.
        dst_size (tuple): The size of the output surface (width, height).

        Returns:
        pygame.Surface: The transformed Pygame surface.
        """
        # Convert Pygame surface to numpy array
        src_array = pygame.surfarray.array3d(surface)

        # Swap axes to match OpenCV format
        src_array = np.swapaxes(src_array, 0, 1)

        x = np.zeros((src_array.shape[0], src_array.shape[1], 4), dtype=np.uint8)
        x += 255
        x[:, :, :3] = src_array

        # Apply perspective warp using OpenCV
        dst_array = cv2.warpPerspective(
            x,
            matrix.astype(float),
            dst_size,
            flags=cv2.INTER_NEAREST,
        )

        a = np.ascontiguousarray(dst_array)
        return pygame.image.frombuffer(a.tobytes(), a.shape[1::-1], "RGBA")

    def transform_point(self, point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Transform a single point using the perspective transformation matrix.

        Parameters:
        point (numpy.ndarray): The point to transform, shape (1, 2).
        matrix (numpy.ndarray): The 3x3 transformation matrix.

        Returns:
        numpy.ndarray: The transformed point, shape (1, 2).
        """
        point = np.array([[point]], dtype=np.float32)  # Shape (1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, matrix)
        return transformed_point[0][0]

    def get_clickable_rect(self) -> pygame.Rect:
        """
        Get the rect of the card where a click in the rect counts as a click
        on the card.
        """
        return pygame.rect.Rect(
            self.position[0] - self.size.x / 2,
            self.position[1] - self.size.y / 2,
            self.size.x,
            self.size.y,
        )

    def process_events(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.get_clickable_rect().collidepoint(event.pos):
                if self.state == self.CardState.IDLE:
                    self.transition_to_grabbed(event)
                    return True

        if event.type == pygame.MOUSEMOTION:
            if self.state == self.CardState.GRABBED:
                self.card_goal_position = pygame.Vector2(event.pos)
                return True

        if event.type == pygame.MOUSEBUTTONUP:
            if self.state == self.CardState.GRABBED:
                self.transition_to_idle()
                return True

        return False

    def transition_to_grabbed(self, event: pygame.event.Event) -> None:
        self.card_pickup_position = self.position
        self.card_goal_position = pygame.Vector2(event.pos)
        self.state = self.CardState.GRABBED
        audio_manager.play_sound("card_sound_5")
        self.tweener_snap_to_mouse.start()

    def transition_to_idle(self) -> None:
        self.state = self.CardState.IDLE
        audio_manager.play_sound("card_sound_4")


class CardController:
    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        self.cards: list[Card] = []

    def add_card(self, card: Card) -> None:
        self.cards.append(card)

    def render(self, surface: pygame.surface, delta: float) -> None:
        for card in reversed(self.cards):
            card.render(surface, delta)

    def process_event(self, event: pygame.event.Event) -> bool:
        for i, card in enumerate(self.cards):
            if card.process_events(event):
                self.cards.insert(0, self.cards.pop(i))
                return True


class Particle:
    def __init__(
        self,
        x: float,
        y: float,
        radius: int,
        color: pygame.Color,
        speed: float,
        lifespan: float,
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.speed = speed
        self.lifespan = lifespan
        self.age = 0.0

    def update(self, delta):
        self.y += self.speed * delta
        self.age += delta

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


class ParticleSystem:
    def __init__(self, x: float, y: float, colours: list[pygame.Color]):
        self.x = x
        self.y = y
        self.particles: list[Particle] = []
        self.spawn_rate = 20
        self.time_since_last_spawn = 0.0
        self.colours = colours

    def update(self, delta: float):
        self.time_since_last_spawn += delta
        # Spawn new particles
        if self.time_since_last_spawn >= 1.0 / self.spawn_rate:
            self.time_since_last_spawn -= 1.0 / self.spawn_rate
            self.particles.append(self.create_particle())

        # Update existing particles
        for particle in self.particles:
            particle.update(delta)
            if particle.age >= particle.lifespan:
                self.particles.remove(particle)

    def create_particle(self):
        radius = random.randint(2, 10)
        color = random.choice(self.colours)
        speed = random.uniform(50, 100)
        lifespan = 3
        return Particle(
            self.x + random.randint(0, 60),
            self.y - radius,
            radius,
            color,
            speed,
            lifespan,
        )

    def render(self, surface: pygame.Surface):
        for particle in self.particles:
            particle.draw(surface)


class AnimatedBackground:

    DARK = pygame.Color(91, 110, 225)
    LIGHT = pygame.Color(99, 155, 255)
    WHITE = pygame.Color(255, 255, 255)

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.surface = pygame.surface.Surface(size)
        self.particle_system = ParticleSystem(0, 0, [self.DARK, self.LIGHT, self.WHITE])

    def render(self, delta: float) -> None:
        self.surface.fill(self.WHITE)
        self.particle_system.update(delta)
        self.particle_system.render(self.surface)

    def get_surface(self) -> pygame.Surface:
        return self.surface


def main() -> None:
    pygame.init()
    pygame.display.set_caption(
        "Card Demo",
    )

    font_manager.load_font("main_font", M5X7, 16)
    audio_manager.load_sound(
        "card_sound_4", os.path.join(get_sounds_directory(), "card_sound_4.wav")
    )
    audio_manager.load_sound(
        "card_sound_5", os.path.join(get_sounds_directory(), "card_sound_5.wav")
    )
    audio_manager.set_sound_volume("card_sound_4", 0.5)
    audio_manager.set_sound_volume("card_sound_5", 0.5)

    screen = pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        flags=pygame.RESIZABLE | pygame.SCALED,
    )

    game_clock = pygame.time.Clock()

    card_background = pygame.image.load(
        os.path.join(get_cards_directory(), "card_1.png")
    )
    card_2_background = pygame.image.load(
        os.path.join(get_cards_directory(), "card_2.png")
    )
    card_3_background = pygame.image.load(
        os.path.join(get_cards_directory(), "card_3.png")
    )
    animated_background = AnimatedBackground((60, 90))

    card = Card(
        position=pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        background_surface=card_background,
        size=pygame.Vector2(60, 90),
    )
    card2 = Card(
        position=pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        background_surface=card_2_background,
        size=pygame.Vector2(60, 90),
    )
    card3 = Card(
        position=pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        background_surface=card_3_background,
        size=pygame.Vector2(60, 90),
    )
    card4 = Card(
        position=pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        background_surface=animated_background.get_surface(),
        size=pygame.Vector2(60, 90),
    )
    card5 = Card(
        position=pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),
        background_surface=card_background,
        size=pygame.Vector2(45, 30),
    )

    card_controller = CardController(screen)
    card_controller.add_card(card)
    card_controller.add_card(card2)
    card_controller.add_card(card3)
    card_controller.add_card(card4)
    card_controller.add_card(card5)

    time = 0

    running = True
    while True:

        delta_ms = game_clock.tick(144)
        delta = delta_ms / 1000
        time += delta

        animated_background.render(delta)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            card_controller.process_event(event)

        screen.fill("white")

        card_controller.render(screen, delta)

        pygame.display.flip()

        if not running:
            break

    pygame.quit()


main()
