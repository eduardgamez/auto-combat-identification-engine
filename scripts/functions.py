import numpy as np

def sort_attributes(df):
    """Sorts the attributes of a DataFrame by their conditional entropy relative to the Target."""
    entropy_var_list = np.array([])
    attributes_list = df.columns[:-1]
    target = df.columns[-1] # Automatically identifies the target column (e.g., 'Target_Friend')

    for var in attributes_list:
        cond_entropy = 0
        # Divide the dataset by each possible value of the attribute
        for val, subset in df.groupby(var):
            # Calculate the entropy of the TARGET in this subset
            probs = (subset[target].value_counts() / len(subset)).to_numpy()
            ent_subset = -np.sum(probs * np.log2(probs, where=(probs > 0), out=np.zeros_like(probs)))
            
            # Add the weighted entropy
            weight = len(subset) / len(df)
            cond_entropy += weight * ent_subset
            
        entropy_var_list = np.append(entropy_var_list, cond_entropy)

    # Sort: the one with the lowest conditional entropy provides the most information
    sorted_indices = np.argsort(entropy_var_list)
    return np.array(attributes_list)[sorted_indices]

def build_tree(df, sorted_attributes):
    """Builds the decision tree in memory (as nested dictionaries) for extreme speed."""
    target = df.columns[-1]
    
    # Base case 1: Pure node (all instances belong to the same class, entropy = 0)
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
        
    # Base case 2: If we run out of attributes to check, return the statistical mode
    if len(sorted_attributes) == 0:
        return df[target].mode()[0]
        
    # Take the first attribute from the prioritized list
    attr = sorted_attributes[0]
    tree = {attr: {}}
    
    # Recursively build branches for each possible value in the data
    for val, subset in df.groupby(attr):
        tree[attr][val] = build_tree(subset, sorted_attributes[1:])
        
    # Save the local mode in case a test instance has a rare value not seen during training
    tree[attr]['__mode__'] = df[target].mode()[0]
    
    return tree

def predict_case(tree, row):
    """Classifies a single row ultra-fast using the prebuilt tree."""
    # If it's not a dictionary, we've reached an answer leaf (0 or 1)
    if not isinstance(tree, dict):
        return tree
        
    # Extract the attribute to check at this node
    attr = list(tree.keys())[0]
    val = row[attr]
    
    # Find the branch corresponding to the row's value
    branch = tree[attr].get(val)
    
    # If the value didn't exist in training, return the mode to prevent failure
    if branch is None:
        return tree[attr]['__mode__']
        
    # Repeat for the next level by navigating down the branch
    return predict_case(branch, row)
