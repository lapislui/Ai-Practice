import AI_data as aid

# Load your data using the same style as the rest of AI_data.
loader = aid.LoadData("data/")
df = loader.read()

# Optional: inspect your DataFrame.
df.info()

# Simple Linear Regression: just provide X and Y column names.
reg = aid.SimpleLinearRegression(df)
reg.fit("experience_years", "salary")

# Results and predictions.
reg.summary()
print("Prediction:", reg.predict(5))
print("Multiple predictions:", reg.predict([1, 3, 5, 8]))

# Optional visualizations.
reg.plot()
reg.loss_plot()
reg.residual_plot()
