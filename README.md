# Logistic Regression from Scratch

## Overview
This project implements **logistic regression from scratch** using gradient descent and compares it against a **least squares classifier**. It focuses on understanding optimization, loss functions, and model robustness in classification tasks.

---

## Key Features

### 🔹 Logistic Regression Implementation
- Implemented binary logistic regression without high-level ML libraries
- Optimized using gradient descent
- Includes L2 regularization

### 🔹 Least Squares Classifier
- Implemented closed-form solution
- Compared performance against logistic regression

### 🔹 Robustness to Outliers
- Added synthetic outliers far from the decision boundary
- Evaluated how each model reacts to these extreme points

### 🔹 Visualization
- Plotted training data and predicted classifications
- Visualized how decision boundaries change

---

## Methods

### Logistic Loss
The model minimizes:
    log(1 + exp(-y xᵀw)) + λ||w||²

### Gradient Descent
Weights are updated iteratively:
    w = w - η ∇L

### Least Squares
Closed-form solution:
    w = (XᵀX + λI)⁻¹ Xᵀy

---

## Results & Insights

- Logistic regression provides more stable performance under outliers  
- Least squares is more sensitive to extreme data points  
- Logistic loss better aligns with classification objectives  
- Both methods perform similarly on clean data  

---

## Technologies Used

- Python
- NumPy
- Matplotlib

---

## How to Run

### 1. Install dependencies
```
pip install numpy matplotlib
```

### 2. Run the script
```
python src/logistic_regression.py
```

---

## Applications

- Binary classification problems
- Understanding optimization in machine learning
- Comparing loss functions
- Studying robustness to noisy data

---


## Summary

This project demonstrates the importance of choosing appropriate loss functions for classification tasks and highlights the robustness advantages of logistic regression compared to least squares under noisy conditions.
