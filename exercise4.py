import gymnasium as gym
from parser import *
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

BASE_SEED = 9902

# --- 1. Load and split (same as before) ---
df = pd.read_csv("experiences.csv")
df = df[df['step'] != 'end'].copy()
for col in ['state', 'action', 'attempt']:
    df[col] = df[col].astype(int)
df['player'] = df['player'].astype(str)

episodes = df[['player', 'attempt']].drop_duplicates()
episodes = episodes.sample(frac=1, random_state=BASE_SEED).reset_index(drop=True)
split_idx = int(len(episodes) * 0.8)

train_df = pd.merge(df, episodes.iloc[:split_idx], on=['player', 'attempt'], how='inner')
X_train = train_df[['state']]   # <-- DataFrame with feature name 'state'
y_train = train_df['action']

default_model = DecisionTreeClassifier(random_state=BASE_SEED).fit(X_train, y_train)
best_model = DecisionTreeClassifier(
    class_weight=None, criterion='gini', max_depth=20,
    min_samples_split=10, splitter='random', random_state=BASE_SEED
).fit(X_train, y_train)

# --- 2. Evaluate ---
def evaluate_ml_agent(model, is_slippery, n_episodes=50):
    env = gym.make(
        "FrozenLake-v1",
        desc=prepare_for_env("map_1.txt"),
        is_slippery=is_slippery,
        max_episode_steps=200,
        success_rate=0.75,
        render_mode=None
    )
    outcomes, lengths = [], {"goal": [], "hole": [], "timeout": []}

    for k in range(n_episodes):
        obs, _ = env.reset(seed=BASE_SEED + 10000 + k)
        step = 0
        while True:
            # FIX: wrap in DataFrame to match training feature names
            action = int(model.predict(pd.DataFrame({'state': [obs]}))[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            if terminated or truncated:
                if terminated and reward == 1:
                    outcomes.append("goal"); lengths["goal"].append(step)
                elif terminated:
                    outcomes.append("hole"); lengths["hole"].append(step)
                else:
                    outcomes.append("timeout"); lengths["timeout"].append(step)
                break
    env.close()

    counts = Counter(outcomes)
    return {
        "success_rate": counts["goal"] / n_episodes * 100,
        "counts": dict(counts),
        "mean_lengths": {k: float(np.mean(v)) if v else 0.0 for k, v in lengths.items()}
    }

if __name__ == "__main__":
    print("=== Default Model ===")
    print(evaluate_ml_agent(default_model, True))

    print("\n=== Best Model ===")
    print(evaluate_ml_agent(best_model, True))