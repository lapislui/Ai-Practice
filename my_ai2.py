import AI_data as aid

# Load data
loader = aid.LoadData("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
df = loader.read()

# Create univariate object
uni = aid.Univariate(df)

# Show all available methods
# uni.help()

# Get help for an individual method
# uni.help("value_counts")

uni.value_counts("")

