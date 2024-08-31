import os

import pygame

from common.file_helpers import get_gems_directory
from common.match3.gem_types import GemTypes


class GemAssets:
    def __init__(self) -> None:
        self.gem_frames: dict[GemTypes, list[pygame.Surface]] = {}

        for gem_type in GemTypes:
            gem_type_frames: list[pygame.Surface] = []
            for i in range(8):
                gem_type_frames.append(
                    pygame.image.load(
                        os.path.join(
                            get_gems_directory(),
                            f"{gem_type.name.lower()}_gem_{i}.png",
                        )
                    )
                )
            self.gem_frames[gem_type] = gem_type_frames

    def get_gem_frames(self, gem_type: GemTypes) -> list[pygame.Surface]:
        return self.gem_frames[gem_type]
