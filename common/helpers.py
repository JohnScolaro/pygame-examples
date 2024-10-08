from collections import deque
from typing import Callable


class Tweener:
    """
    An object to handle everything tweening.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        duration_seconds: float,
        tween_function: None | Callable[[float], float] = None,
    ):
        """
        A tweener object for tweening things. Stores the state of the tween
        and updates itself each frame.
        """
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration_seconds
        if tween_function is not None:
            self.tween_function = tween_function
        else:
            self.tween_function = lambda x: x

        self.elapsed = 0
        self.value = end_value
        self.finished = True

    def update(self, delta_time: float) -> None:
        if self.finished:
            return

        self.elapsed += delta_time
        if self.elapsed >= self.duration:
            self.elapsed = self.duration
            self.finished = True

        t = self.elapsed / self.duration
        self.value = self.tween_function(
            self.start_value + (self.end_value - self.start_value) * t
        )
        return self.value

    def start(
        self,
        start_value: None | float = None,
        end_value: None | float = None,
        duration_seconds: None | float = None,
    ) -> None:
        """
        This starts the tween if it's not currently running. It restarts it if
        it is. If any of the optional parameters are specified, then they
        replace the current values for start/end/duration.
        """
        self.finished = False
        self.elapsed = 0

        if start_value is not None:
            self.start_value = start_value
        if end_value is not None:
            self.end_value = end_value
        if duration_seconds is not None:
            self.duration_seconds = duration_seconds

    def get_value(self) -> float:
        return self.value

    def is_finished(self) -> bool:
        return self.finished


class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque(maxlen=size)
        self.sum = 0

    def add(self, value: float) -> None:
        if len(self.queue) == self.size:
            self.sum -= self.queue[0]
        self.queue.append(value)
        self.sum += value

    def average(self) -> float:
        if not self.queue:
            return 0
        return self.sum / len(self.queue)


def wrapf(value: float, min_value: float, max_value: float):
    range_size = max_value - min_value
    wrapped_value = (value - min_value) % range_size + min_value
    return wrapped_value if wrapped_value != max_value else min_value


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def scale_and_clamp(
    in_a: float, in_b: float, out_a: float, out_b: float, value: float
) -> float:
    if in_a == in_b:
        raise ValueError("Both inputs cannot be the same value")

    # Scale the value
    scaled_value = ((value - in_a) / (in_b - in_a)) * (out_b - out_a) + out_a

    # Clamp the value within the output range
    clamped_value = clamp(scaled_value, min(out_a, out_b), max(out_a, out_b))

    return clamped_value
