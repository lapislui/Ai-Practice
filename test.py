import Ai_data as aid

# Load dataset
loader = aid.LoadData("polling_on_ai_usage_jul_2026.csv")
df = loader.read()

# Create data definition object
data = aid.DataDefinition(df)

# Test special methods
print(data)                   # Dataset summary
# print(len(data))              # Number of rows
# print(repr(data))             # Developer representation
# print(data.columns())         # Column names
# print(data.columns()[0] in data)  # Check first column

# Run individual checks when needed
# print(data.shape())
print(data.sample(5))
# print(data.data_types())
# data.info()
# print(data.describe())
# print(data.missing_values())
# print(data.missing_percentage())
# print(data.duplicates())
# print(data.unique_values())
# print(data.memory_usage())


# Create univariate object
uni = aid.Univariate(df)

# Show all available methods
uni.help()

# Get help for an individual method
uni.help("value_counts")
uni.help("skewness")

# Statistical analysis
print(uni.value_counts("Sex"))
print(uni.summary("Age"))
print(uni.minimum("Age"))
print(uni.maximum("Age"))
print(uni.mean("Age"))
print(uni.median("Age"))
print(uni.skewness("Age"))

# Visualizations
uni.countplot("Embarked")
uni.pie_chart("Sex")
uni.histogram("Age")
uni.boxplot("Age")

bi = aid.Bivariate(df)

# Help for all methods
bi.help()

# Help for individual methods
bi.help("scatterplot")
bi.help("heatmap")
bi.help("correlation")

# Numerical vs numerical
print(bi.correlation())
bi.scatterplot("Age", "Fare")

# Numerical vs categorical
bi.barplot("Sex", "Age")
bi.boxplot("Sex", "Age")
bi.kdeplot("Age", hue="Survived")

# Categorical vs categorical
print(bi.crosstab("Pclass", "Survived"))
bi.heatmap("Pclass", "Survived")

# Multiple numerical columns
bi.pairplot(columns=["Age", "Fare"])

# Grouped analysis
print(bi.group_summary("Sex", "Age"))




bi = aid.Bivariate(df)

# Help for all methods
bi.help()

# Help for individual methods
bi.help("scatterplot")
bi.help("heatmap")
bi.help("correlation")

# Numerical vs numerical
print(bi.correlation())
bi.scatterplot("Age", "Fare")

# Numerical vs categorical
bi.barplot("Sex", "Age")
bi.boxplot("Sex", "Age")
bi.kdeplot("Age", hue="Survived")

# Categorical vs categorical
print(bi.crosstab("Pclass", "Survived"))
bi.heatmap("Pclass", "Survived")

# Multiple numerical columns
bi.pairplot(columns=["Age", "Fare"])

# Grouped analysis
print(bi.group_summary("Sex", "Age"))