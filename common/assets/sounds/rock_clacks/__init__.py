import os

NUM_ROCK_CLACKS_SOUNDS = 25
ROCK_CLACKS = list(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"rocks_clacking_{i:02d}.wav"
    )
    for i in range(NUM_ROCK_CLACKS_SOUNDS)
)
