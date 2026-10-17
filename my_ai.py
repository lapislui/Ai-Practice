import AI_data as aid

loader = aid.LoadData("polling_on_ai_usage_jul_2026.csv")
df = loader.read()

# loader.help()
# loader.help("read")

data = aid.DataDefinition(df)

# data.help()
# data.help("shape")
# data.help("sample")

print(data.shape())
print(data.sample(3))
print(data.info())

# uni = aid.Univariate(df)

# uni.help()