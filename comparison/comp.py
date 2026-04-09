import sys
import os
import pandas as pd

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from scripts.functions import sort_attributes, build_tree, predict_case

# 1. Load data
train, test = pd.read_csv('dataset/dataset.csv'), pd.read_csv('test/test_cases.csv')
target = train.columns[-1]

# 2. Train the tree
tree = build_tree(train, sort_attributes(train))

# 3. Classification
preds_tree = test.apply(lambda r: predict_case(tree, r), axis=1)
preds_iff = test['IFF_Response'].apply(lambda x: 1 if x == 'Valid' else 0)

# 4. Accuracy Calculation
acc_tree = (preds_tree == test[target]).mean()
acc_iff = (preds_iff == test[target]).mean()

print(f"Accuracy IFF ONLY: {acc_iff:.2%}")
print(f"Accuracy ID3 TREE: {acc_tree:.2%}")
print(f"Tree improvement: {acc_tree - acc_iff:.2%}")
