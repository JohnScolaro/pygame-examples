import os

COMBO_SOUNDS = list(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f"combo_sound_{i:02d}.wav")
    for i in range(16)
)
