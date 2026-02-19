# Simple Decision Tree ML (Educational Project)

## Overview

This is a **personal, educational project** aimed at learning the fundamentals of machine learning and decision tree algorithms. It implements a **custom Decision Tree classifier** in Python, built from scratch.  

⚠️ **Important:** This project is **highly inefficient**, may contain bugs, and is intended **only for learning purposes**.  

Currently, the model works only with **tabular datasets formatted as `[feature1, feature2, ..., featureN, label]`** from sklearn library, such as:

- `digits`  
- `breast_cancer`  
- `wine`  

Support for additional datasets may be added in the future.

---

## Algorithm

- **Type:** Decision Tree Classifier (supervised learning)  
- **Splitting criterion:** Gini impurity  
- **Method:**  
  - Evaluates all features and thresholds (step size configurable)  
  - Chooses the split with the lowest weighted Gini  
  - Recursively creates child nodes until leaves are pure or no better split is possible  
- **Leaves:** Store the class of the group to make predictions  
- **Prediction:** Traverses the tree from root to leaf and returns the class at the leaf  

---

## Limitations

- Works only on **preformatted tabular datasets**.  
- Inefficient compared to `sklearn`’s `DecisionTreeClassifier`.  
- Step size and leaf size affect performance and accuracy.  
- Accuracy may be overestimated due to overfitting on small datasets.  

---

## Dependencies

- **Python 3.x**  
- **NumPy** – for array operations  
- **anytree** – for storing and visualizing the tree  
- **scikit-learn** – only for accessing datasets  
