import sys
import os
import pandas as pd

# Add the root directory to the path to import from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.functions import sort_attributes, build_tree, predict_case

def main():
    print("1. Loading datasets...")
    train_df = pd.read_csv('dataset/dataset.csv')
    test_df = pd.read_csv('test/test_cases.csv')
    
    print("2. Calculating entropy and sorting attributes...")
    sorted_attrs = sort_attributes(train_df)
    print(f"Order: {sorted_attrs}")
    
    print("3. Building the tree in memory (training phase)...")
    tree = build_tree(train_df, sorted_attrs)
    
    print("4. Evaluating test cases (ultra-fast execution phase)...")
    correct = 0
    total = len(test_df)
    target = train_df.columns[-1]
    
    # Separate the attributes from the real response
    features_test = test_df.drop(columns=[target])
    real_values = test_df[target].to_numpy()
    
    # Here is where we use predict_case massively but case by case
    for i, row in features_test.iterrows():
        prediction = predict_case(tree, row)
        if prediction == real_values[i]:
            correct += 1
            
    accuracy = (correct / total) * 100
    print(f"\nResults: {correct}/{total} correct ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()
