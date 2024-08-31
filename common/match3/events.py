import dataclasses

import pygame

from common.match3.backend_puzzle import Match
from common.match3.gem_types import GemTypes

SCORING_EVENT = pygame.event.custom_type()


@dataclasses.dataclass
class ScoreEvent:
    scoring_matches: list[tuple[Match, GemTypes]]
    combo_level: int


def post_scoring_event(score_event: ScoreEvent) -> None:
    event = create_event(score_event)
    success = pygame.event.post(event)
    if not success:
        raise Exception(f'Posting event "{event}" went wrong somehow?')


def create_event(score_event: ScoreEvent) -> pygame.Event:
    return pygame.Event(SCORING_EVENT, score_event=score_event)
