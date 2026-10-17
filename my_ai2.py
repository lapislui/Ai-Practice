import AI_data as aid

# Load data
loader = aid.LoadData("polling_on_ai_usage_jul_2026.csv")
df = loader.read()

# Create univariate object
uni = aid.Univariate(df)

# Show all available methods
# uni.help()

# Get help for an individual method
# uni.help("value_counts")
 
# uni.value_counts("")

data.info()