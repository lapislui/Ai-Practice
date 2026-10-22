import AI_data as aid

# Load data
loader = aid.LoadData("employee_data_practice.csv")
df = loader.read()

# Create univariate object
uni = aid.Univariate(df)

# Show all available methods
# uni.help()

# Get help for an individual method
# uni.help("value_counts")
 
# uni.value_counts("")

df.info()