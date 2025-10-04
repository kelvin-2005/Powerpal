# core/bandit.py
import random
from typing import List, Dict, Tuple

def recommend(actions: List[Dict], expected_rewards: List[float], eps: float = 0.1) -> int:
    """
    actions: list of action dicts (for display)
    expected_rewards: same length; AED saved or utility
    eps-greedy: with prob eps choose random; else choose argmax
    returns index of chosen action
    """
    assert len(actions) == len(expected_rewards)
    if random.random() < eps:
        return random.randrange(len(actions))
    return max(range(len(actions)), key=lambda i: expected_rewards[i])
