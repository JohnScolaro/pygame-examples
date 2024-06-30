import pygame


class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.sfx_muted = False
        self.sfx_volume = 1.0

    def load_sound(self, name, filepath):
        sound = pygame.mixer.Sound(filepath)
        self.sounds[name] = sound

    def play_sound(self, name):
        channel = pygame.mixer.find_channel()
        channel.set_volume(self.sfx_volume)
        if not self.sfx_muted:
            self.sounds[name].play()

    def set_sound_volume(self, name, volume):
        self.sounds[name].set_volume(volume)

    def get_sound_volume(self, name) -> float:
        return self.sounds[name].get_volume()

    def set_sfx_volume(self, volume: float) -> None:
        self.sfx_volume = volume

    def mute_sfx(self) -> None:
        self.set_sfx_volume(0.0)
        self.sfx_muted = True

    def unmute_sfx(self) -> None:
        self.sfx_muted = False
        self.set_sfx_volume(self.sfx_volume)

    def toggle_sfx_mute(self) -> None:
        if self.sfx_muted:
            self.unmute_sfx()
        else:
            self.mute_sfx()

    def stop_sound(self, name):
        self.sounds[name].stop()

    def load_music(self, filepath):
        pygame.mixer.music.load(filepath)
        self.music_loaded = True

    def play_music(self, loops=-1):
        if self.music_loaded:
            pygame.mixer.music.play(loops)
        else:
            raise Exception("No music loaded!")

    def pause_music(self):
        pygame.mixer.music.pause()

    def unpause_music(self):
        pygame.mixer.music.unpause()

    def stop_music(self):
        pygame.mixer.music.stop()

    def set_music_volume(self, volume: float):
        pygame.mixer.music.set_volume(volume)

    def get_music_volume(self) -> float:
        return pygame.mixer.music.get_volume()
