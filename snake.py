"""A compact Snake environment with a safe imitation-learning teacher."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable

Vec = tuple[int, int]
DIRECTIONS: tuple[Vec, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES = ("left", "straight", "right")


def add(a: Vec, b: Vec) -> Vec:
    return a[0] + b[0], a[1] + b[1]


@dataclass
class SnakeGame:
    width: int = 18
    height: int = 14
    seed: int = 4
    body: list[Vec] = field(default_factory=list)
    direction: int = 1
    food: Vec | None = None
    score: int = 0
    alive: bool = True

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> None:
        mid = (self.width // 2, self.height // 2)
        self.body = [mid, (mid[0] - 1, mid[1]), (mid[0] - 2, mid[1])]
        self.direction, self.score, self.alive = 1, 0, True
        self._spawn_food()

    def _spawn_food(self) -> None:
        spaces = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.body]
        self.food = self.rng.choice(spaces) if spaces else None

    def _heading(self, action: int) -> int:
        return (self.direction + (action - 1)) % 4

    def would_collide(self, action: int) -> bool:
        head = add(self.body[0], DIRECTIONS[self._heading(action)])
        return head[0] < 0 or head[0] >= self.width or head[1] < 0 or head[1] >= self.height or head in self.body[:-1]

    def sensory_state(self) -> tuple[int, tuple[bool, bool, bool]]:
        assert self.food is not None
        hx, hy = self.body[0]
        fx, fy = self.food
        forward = DIRECTIONS[self.direction]
        left = DIRECTIONS[(self.direction - 1) % 4]
        lateral = (fx - hx) * left[0] + (fy - hy) * left[1]
        ahead = (fx - hx) * forward[0] + (fy - hy) * forward[1]
        relation = -1 if lateral > 0 else 1 if lateral < 0 else 0 if ahead >= 0 else -1
        return relation, (self.would_collide(0), self.would_collide(1), self.would_collide(2))

    def teacher_action(self) -> int:
        relation, danger = self.sensory_state()
        preferred = relation + 1
        if not danger[preferred]:
            return preferred
        for action in (1, 0, 2):
            if not danger[action]:
                return action
        return 1

    def step(self, action: int) -> dict:
        if not self.alive:
            return self.snapshot()
        self.direction = self._heading(action)
        head = add(self.body[0], DIRECTIONS[self.direction])
        if self.would_collide(1):
            self.alive = False
            return self.snapshot()
        self.body.insert(0, head)
        if head == self.food:
            self.score += 1
            self._spawn_food()
        else:
            self.body.pop()
        return self.snapshot()

    def snapshot(self) -> dict:
        return {"width": self.width, "height": self.height, "body": self.body, "food": self.food, "direction": self.direction, "score": self.score, "alive": self.alive}
