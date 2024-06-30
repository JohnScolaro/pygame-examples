import pygame


class FontManager:
    def __init__(self):
        pygame.font.init()
        self.fonts = {}

    def load_font(self, name, filepath, size):
        """Load a font and store it in the fonts dictionary."""
        try:
            font = pygame.font.Font(filepath, size)
            self.fonts[name] = font
        except pygame.error as e:
            print(f"Failed to load font {filepath}: {e}")

    def get_font(self, name):
        """Retrieve a font by name."""
        if name in self.fonts:
            return self.fonts[name]
        else:
            print(f"Font {name} not found!")
            return None
