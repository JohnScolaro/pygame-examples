import os

NUM_GLASS_CLINKS = 3
GLASS_CLINKS = list(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), f"glass_clink_{i}.wav")
    for i in range(NUM_GLASS_CLINKS)
)
