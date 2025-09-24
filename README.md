# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Pelleti Sindhu Sri
RegisterNumber: 212224240113
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.DataFrame({
    'Population': [6.1101, 5.5277, 8.5186, 7.0032, 5.8598,
                   8.3829, 7.4764, 8.5781, 6.4862, 5.0546],
    'Profit': [17.592, 9.1302, 13.662, 11.854, 6.8233,
               11.886, 4.3483, 12.0, 6.5987, 3.8166]
})



print(data.head())


plt.scatter(data['Population'], data['Profit'], marker='x', color='red')
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit (10,000s $)")
plt.title("Profit vs Population")
plt.show()


X = data['Population'].values
y = data['Profit'].values
m = len(y)


X = np.c_[np.ones(m), X]


theta = np.zeros(2)


def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1/(2*m)) * np.dot(error.T, error)
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        theta = theta - (alpha/m) * X.T.dot(error)
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history


alpha = 0.01
iterations = 1500
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("Final Theta values:", theta)
print("Final Cost:", cost_history[-1])


plt.plot(range(iterations), cost_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function Convergence")
plt.show()


plt.scatter(data['Population'], data['Profit'], marker='x', color='red')
plt.plot(data['Population'], X.dot(theta), color='blue')
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit (10,000s $)")
plt.title("Linear Regression Fit")
plt.show()


predict1 = np.dot([1, 3.5], theta)   
predict2 = np.dot([1, 7], theta)     

print("For population = 35,000, predicted profit = $", predict1*10000)
print("For population = 70,000, predicted profit = $", predict2*10000)

```

## Output:

<img width="747" height="736" alt="Screenshot 2025-09-24 215338" src="https://github.com/user-attachments/assets/fd10f2bf-1060-4ccb-947e-c7b5b3a79fb8" />

<img width="742" height="551" alt="Screenshot 2025-09-24 215350" src="https://github.com/user-attachments/assets/7a7380d3-8c49-42e2-b0c6-739a248280d1" />

<img width="759" height="602" alt="Screenshot 2025-09-24 215402" src="https://github.com/user-attachments/assets/5c7feed4-37de-44a5-ae47-8c0c371e6b31" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
