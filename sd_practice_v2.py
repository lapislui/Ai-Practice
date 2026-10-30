import AI_data as aid

# Step 1: Load data using our module
loader = aid.LoadData("employee_data_practice.csv")
df = loader.read()

# Step 2: Select features and target
X = df[["Age", "ExperienceYears"]]
y = df["AnnualSalaryINR"]

# Step 3: Create our standardization object
sd = aid.StandardDeviation()

# Step 4: Split before standardization
X_train, X_test, y_train, y_test = sd.train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

# Step 5: Learn statistics from training data only
sd.fit(X_train)

# Step 6: Standardize both datasets using the same statistics
X_train_scaled = sd.transform(X_train)
X_test_scaled = sd.transform(X_test)

# Step 7: View the results
print("Training data:")
print(X_train_scaled.head())

print("\nTesting data:")
print(X_test_scaled.head())

print("\nTraining means:")
print(sd.mean())

print("\nTraining standard deviations:")
print(sd.standard_deviation())