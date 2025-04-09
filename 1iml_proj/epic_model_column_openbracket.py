import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv").drop(columns=["Id"])

features = train_data.drop(columns=["y", "Id"])
target = train_data["y"]

# train model
model = LinearRegression()
model.fit(features, target)

target_pred = model.predict(features)
error = mean_squared_error(target, target_pred) ** 0.5

# predict target for test-features
target_pred_test = model.predict(test_data)

output = []
for (i, d) in enumerate(target_pred_test):
	output.append((i + 10000, float(d)))

print("output:", output)
pd.DataFrame(data = output).to_csv("output.csv", index = False, header = ["Id", "y"])