import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

BASE_SEED = 9902

#  1. Load data 
df = pd.read_csv("experiences.csv")
df = df[df['step'] != 'end'].copy()
df['state']  = df['state'].astype(int)
df['action'] = df['action'].astype(int)
df['attempt']= df['attempt'].astype(int)
df['player'] = df['player'].astype(str)

episodes = df[['player', 'attempt']].drop_duplicates()
episodes = episodes.sample(frac=1, random_state=BASE_SEED).reset_index(drop=True)

split_idx = int(len(episodes) * 0.8)
train_episodes = episodes.iloc[:split_idx]
test_episodes  = episodes.iloc[split_idx:]

train_df = pd.merge(df, train_episodes, on=['player', 'attempt'], how='inner')
test_df  = pd.merge(df, test_episodes,  on=['player', 'attempt'], how='inner')

X_train, y_train = train_df[['state']], train_df['action']
X_test,  y_test  = test_df[['state']],  test_df['action']

print("Train class distribution:\n", y_train.value_counts().sort_index())
print("\nTest class distribution:\n",  y_test.value_counts().sort_index())

#  2. Grid 
param_grid = {
    'criterion':         ['gini', 'entropy'],
    'splitter':          ['best', 'random'],
    'class_weight':      [None, 'balanced'],
    'max_depth':         [5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 20],
}

#  3. FIXED Scorer 
# Use a plain callable so sklearn's pos_label logic is bypassed.
def macro_f1_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    # Only score on classes present in y OR the model's classes
    labels = sorted(set(y) | set(y_pred))
    return f1_score(y, y_pred, average='macro', labels=labels, zero_division=0)

scoring = {'macro_f1': macro_f1_scorer, 'accuracy': 'accuracy'}

#  4. Grid Search 
clf = DecisionTreeClassifier(random_state=BASE_SEED)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=BASE_SEED)

grid_search = GridSearchCV(clf, param_grid, cv=cv,
                           scoring=scoring, refit='macro_f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 5. Top 10 
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.sort_values('mean_test_macro_f1', ascending=False).head(10)

print("\n=== Top 10 Parameter Combinations (Cross-Validation) ===")
for i, (_, row) in enumerate(top_10.iterrows(), 1):
    print(f"Rank {i}: {row['params']}")
    print(f"   Macro-F1: {row['mean_test_macro_f1']:.4f} "
          f"(±{row['std_test_macro_f1']:.4f}) | "
          f"Accuracy: {row['mean_test_accuracy']:.4f} "
          f"(±{row['std_test_accuracy']:.4f})")

#  6. Default model on test set 
default_clf = DecisionTreeClassifier(random_state=BASE_SEED).fit(X_train, y_train)
d_pred = default_clf.predict(X_test)
d_labels = sorted(set(y_test) | set(d_pred))
print("\n=== Default Model Test Performance ===")
print(f"Accuracy: {accuracy_score(y_test, d_pred):.4f} | "
      f"Macro-F1: {f1_score(y_test, d_pred, average='macro', labels=d_labels, zero_division=0):.4f}")

#  7. Best model on test set 
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
labels = sorted(set(y_test) | set(y_pred))
print("\n=== Best Model Test Set Evaluation ===")
print("Best Parameters:", grid_search.best_params_)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f} | "
      f"Test Macro-F1: {f1_score(y_test, y_pred, average='macro', labels=labels, zero_division=0):.4f}")
print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
print("Confusion Matrix (labels present in data):")
print(confusion_matrix(y_test, y_pred, labels=labels))

# 8. Tree image 
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=['state'],
          class_names=['Left(0)', 'Up(3)'] if len(best_model.classes_) == 2 else
                      [f'Action{c}' for c in best_model.classes_],
          filled=True, rounded=True, fontsize=8)
plt.savefig('tree_best.png', dpi=150, bbox_inches='tight')
print("\nSaved tree_best.png")