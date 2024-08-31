import random
from enum import Enum, auto

import pygame


class GemTypes(Enum):
    BLUE = auto()
    GREEN = auto()
    RED = auto()
    BLACK = auto()
    ORANGE = auto()
    WHITE = auto()

    @classmethod
    def get_color_from_gem_type(cls, gem_type: "GemTypes") -> pygame.Color:
        return {
            GemTypes.BLUE: pygame.Color(77, 155, 230),
            GemTypes.GREEN: pygame.Color(30, 188, 115),
            GemTypes.RED: pygame.Color(240, 79, 120),
            GemTypes.BLACK: pygame.Color(46, 34, 47),
            GemTypes.ORANGE: pygame.Color(249, 194, 43),
            GemTypes.WHITE: pygame.Color(255, 255, 255),
        }[gem_type]

    @classmethod
    def get_random_gem_type(cls) -> "GemTypes":
        return random.choice([x for x in cls])
