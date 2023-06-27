
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

# Read data from CSV
df = pd.read_csv("data.csv")
X = df["X"].values.reshape(-1, 1)
y = df["y"].values

# Perform Elastic Net Regression
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X, y)
y_pred = enet.predict(X)

# Visualize the results
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, y_pred, color="red", label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Elastic Net Regression")
plt.show()
