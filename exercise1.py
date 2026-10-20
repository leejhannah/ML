import gymnasium as gym
from parser import *
import numpy as np
from collections import Counter

BASE_SEED = 9902

with open("map_1.txt", "r") as f:
    raw_map = [line.strip() for line in f if line.strip()]


def state_to_pos(state): return state // 12, state % 12


def pos_to_state(row, col): return row * 12 + col

def get_tile(r, c):
    if 0 <= r < 12 and 0 <= c < 12:
        return raw_map[r][c]
    return "W"  # W 代表 Wall 墙


MOVES = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def rule_agent_v1(state):
    r, c = state_to_pos(state)
    best_action = 0
    min_dist = float('inf')

    for action, (dr, dc) in MOVES.items():
        nr, nc = r + dr, c + dc
        tile = get_tile(nr, nc)

        if tile == 'H' or tile == 'W':
            continue

        dist = abs(nr - 0) + abs(nc - 0)
        if dist < min_dist:
            min_dist = dist
            best_action = action

    return best_action


def rule_agent_v2(state):
    r, c = state_to_pos(state)
    best_action = 0
    min_score = float('inf')

    for action, (dr, dc) in MOVES.items():
        nr, nc = r + dr, c + dc
        tile = get_tile(nr, nc)

        if tile == 'H' or tile == 'W':
            continue


        risk = 0
        if action in (0, 2):
            if get_tile(r - 1, c) == 'H': risk += 1
            if get_tile(r + 1, c) == 'H': risk += 1
        else:
            if get_tile(r, c - 1) == 'H': risk += 1
            if get_tile(r, c + 1) == 'H': risk += 1


        score = abs(nr - 0) + abs(nc - 0) + risk * 5

        if score < min_score:
            min_score = score
            best_action = action

    return best_action


def evaluate_agent(agent_func, is_slippery, n_episodes=50):
    env = gym.make(
        "FrozenLake-v1",
        desc=prepare_for_env("map_1.txt"),
        is_slippery=is_slippery,
        max_episode_steps=200,
        success_rate=0.75,
        render_mode=None
    )

    outcomes = []
    lengths = {"goal": [], "hole": [], "timeout": []}
    hole_locations = []

    for k in range(n_episodes):
        obs, info = env.reset(seed=BASE_SEED + 10000 + k)
        step = 0
        while True:
            action = agent_func(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                if terminated:
                    if reward == 1:
                        outcomes.append("goal")
                        lengths["goal"].append(step)
                    else:
                        outcomes.append("hole")
                        lengths["hole"].append(step)
                        hole_locations.append(obs)
                else:
                    outcomes.append("timeout")
                    lengths["timeout"].append(step)
                break
    env.close()

    counts = Counter(outcomes)
    success_rate = counts["goal"] / n_episodes * 100

    mean_lengths = {k: np.mean(v) if v else 0 for k, v in lengths.items()}
    most_common_hole = Counter(hole_locations).most_common(1)[0][0] if hole_locations else None

    return {
        "success_rate": success_rate,
        "counts": counts,
        "mean_lengths": mean_lengths,
        "most_common_hole": most_common_hole
    }


if __name__ == "__main__":
    print("Agent 1 (Simple) ")
    for slip in [False, True]:
        res = evaluate_agent(rule_agent_v1, slip)
        print(
            f"Slippery={slip}: Success {res['success_rate']:.1f}% | Ends: {dict(res['counts'])} | MeanLen: {res['mean_lengths']} | Max Hole: {res['most_common_hole']}")

    print("\n Agent 2 (Anti-slip) ")
    for slip in [False, True]:
        res = evaluate_agent(rule_agent_v2, slip)
        print(
            f"Slippery={slip}: Success {res['success_rate']:.1f}% | Ends: {dict(res['counts'])} | MeanLen: {res['mean_lengths']} | Max Hole: {res['most_common_hole']}")
