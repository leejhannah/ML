import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_SEED = 9902

df = pd.read_csv("experiences.csv")
df = df[df['step'] != 'end'].copy()
df['state'] = df['state'].astype(int)
df['action'] = df['action'].astype(int)
df['attempt'] = df['attempt'].astype(int)
df['player'] = df['player'].astype(str)

episodes = df[['player', 'attempt']].drop_duplicates()

episodes = episodes.sample(frac=1, random_state=BASE_SEED).reset_index(drop=True)

split_idx = int(len(episodes) * 0.8)
train_episodes = episodes.iloc[:split_idx]
test_episodes = episodes.iloc[split_idx:]

train_df = pd.merge(df, train_episodes, on=['player', 'attempt'], how='inner')
test_df = pd.merge(df, test_episodes, on=['player', 'attempt'], how='inner')

X_train = train_df[['state']]
y_train = train_df['action']
X_test = test_df[['state']]
y_test = test_df['action']

print(f"Training episodes: {len(train_episodes)}, Test episodes: {len(test_episodes)}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

clf = DecisionTreeClassifier(random_state=BASE_SEED)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

test_correct = (y_test == y_test_pred).sum()
test_incorrect = (y_test != y_test_pred).sum()

print("=== Evaluation Results ===")
print(f"Test set correct predictions: {test_correct}")
print(f"Test set incorrect predictions: {test_incorrect}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}\n")

print("=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_test_pred, zero_division=0))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold

def get_tree_depth(clf):
    if clf.tree_.node_count == 1:
        return 0
    max_depth = 0
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        max_depth = max(max_depth, depth)

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
    return max_depth

depth = get_tree_depth(clf)
n_leaves = (children_left == -1).sum()

print(f"\n=== Tree Structure ===")
print(f"Tree Depth: {depth}")
print(f"Number of Nodes: {n_nodes}")
print(f"Number of Leaves: {n_leaves}")

print("\n=== First Levels of the Tree ===")


def print_tree_rules(clf, max_depth=3):
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    stack = [(0, 0, "")]
    while stack:
        node_id, depth, prefix = stack.pop()

        if depth > max_depth:
            continue

        if children_left[node_id] == -1:
            value = clf.tree_.value[node_id][0]
            action = np.argmax(value)
            print(f"{prefix} -> Predict Action {action}")
        else:

            print(f"{prefix} if state <= {threshold[node_id]:.1f}:")
            stack.append((children_left[node_id], depth + 1, prefix + "  |-- True "))
            stack.append((children_right[node_id], depth + 1, prefix + "  |-- False"))


print_tree_rules(clf, max_depth=3)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=['state'], class_names=['Left(0)', 'Down(1)', 'Right(2)', 'Up(3)'],
          filled=True, rounded=True, max_depth=3, fontsize=10)
plt.savefig('tree_default_depth3.png', dpi=150)
print("\nTree image saved as 'tree_default_depth3.png' (in the current 2_1 folder).")