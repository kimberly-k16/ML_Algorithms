class Node:
    def __init__(self, feature=None, threshold=None, left_tree=None, right_tree=None, value=None):  
        # Feature that this node splits on (None for leaf nodes)
        self.feature = feature
        # Threshold for the split (None for leaf nodes)
        self.threshold = threshold
        # Reference to the left subtree
        self.left_tree = left_tree
        # Reference to the right subtree
        self.right_tree = right_tree
        # Value of the leaf node (either class label for classification or prediction for regression)
        self.value = value

import numpy as np

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, max_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        # Determine max features if not set
        self.max_features = X.shape[1] if not self.max_features else min(X.shape[1], self.max_features)
        # Build the tree
        self.root = self.helper_grow_tree(X, y, 0)
    
    def predict(self, X):
        predictions = [self.traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
    def traverse_tree(self, x, node):
        # Traverse tree until a leaf node is reached
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left_tree)
        return self.traverse_tree(x, node.right_tree)

    def helper_grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        
        # Randomly select a subset of features so all trees look differently
        feature_idxs = np.random.choice(n_features, self.max_features, replace=False)
        
        # Find the best feature and threshold
        best_feature, best_threshold = self.best_split(X, y, feature_idxs)
        
        # Split the data
        left_idxs, right_idxs = self.split(X[:, best_feature], best_threshold)
        left_tree = self.helper_grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_tree = self.helper_grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left_tree, right_tree)

    def best_split(self, X, y, feature_idxs):
        best_gain = -float('inf')  # Start with the worst gain
        split_idx, split_threshold = None, None
        
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # Split data
                left_idxs, right_idxs = self.split(X_column, threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                # Calculate information gain
                gain = self.information_gain(y, left_idxs, right_idxs)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold
        
        return split_idx, split_threshold

    def split(self, X_column, split_threshold):
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def information_gain(self, y, left_idxs, right_idxs):
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0
        
        # Compute weighted average of children's entropy
        e_left, e_right = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Return the information gain
        return parent_entropy - child_entropy

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def most_common_label(self, y):
        return np.bincount(y).argmax()
    
class DecisionTreeRegressor(DecisionTree):
    def __init__(self, min_samples_split=2, max_depth=100, max_features=None):
        super().__init__(min_samples_split, max_depth, max_features)

    def most_common_label(self, y):
        return np.mean(y)

    def information_gain(self, y, left_idxs, right_idxs):
        # Mean Squared Error (MSE)
        parent_mse = self.mean_squared_error(y)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0
        
        # Compute weighted average of children's MSE
        mse_left = self.mean_squared_error(y[left_idxs])
        mse_right = self.mean_squared_error(y[right_idxs])
        child_mse = (n_left / n) * mse_left + (n_right / n) * mse_right
        
        return parent_mse - child_mse

    def mean_squared_error(self, y):
        return np.mean((y - np.mean(y)) ** 2)