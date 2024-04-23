import numpy as np
from datetime import date, timedelta

from dataclasses import dataclass

NEPOCHS = 10
NEMITTERS = 100
EPOCHS_PER_CHUNK = 2

EMITTER_DTYPE = np.dtype(
    [
        ("xpos", "f4"),
        ("ypos", "f4"),
        ("zpos", "f4"),
        ("xvel", "f4"),
        ("yvel", "f4"),
        ("zvel", "f4"),
        ("cbias", "f4"),
        ("cdrift", "f4"),
    ]
)

states_chunk = []

for epoch in range(NEPOCHS):
    epoch_states = [
        tuple(np.random.randn(8).astype(np.float32)) for _ in range(NEMITTERS)
    ]
    epoch_states = np.rec.array(epoch_states, dtype=EMITTER_DTYPE)

    states_chunk.append(epoch_states)

    if epoch % EPOCHS_PER_CHUNK == 0:
        np.save("test", epoch_states)

        states_chunk = []


sim_states = np.load("test.npy", mmap_mode="c")

print()
