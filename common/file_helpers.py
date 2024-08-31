import os


def get_repo_root() -> str:
    # Get the full path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory name of the current file
    current_directory = os.path.dirname(current_file_path)

    return os.path.abspath(os.path.join(current_directory, ".."))


def get_game_directory() -> str:
    return os.path.join(get_repo_root(), "common")


def get_assets_directory() -> str:
    return os.path.join(get_game_directory(), "assets")


def get_sounds_directory() -> str:
    return os.path.join(get_assets_directory(), "sounds")


def get_fonts_directory() -> str:
    return os.path.join(get_assets_directory(), "fonts")


def get_cards_directory() -> str:
    return os.path.join(get_assets_directory(), "cards")


def get_sprites_directory() -> str:
    return os.path.join(get_assets_directory(), "sprites")


def get_gems_directory() -> str:
    return os.path.join(get_sprites_directory(), "gems")
