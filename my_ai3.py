import AI_data as aid

loader = aid.LoadData("penguins.csv")
df = loader.read()
data = aid.DataDefinition(df)

bi = aid.Bivariate(df)
data.info()

# Help for all methods
bi.help()

# Help for individual methods
# bi.help("scatterplot")
# bi.help("heatmap")
bi.help("correlation")

# Numerical vs numerical
# print(bi.correlation())
# bi.scatterplot("flipper_length_mm", "body_mass_g")

# # Numerical vs categorical
# bi.barplot("species", "Age")
# bi.boxplot("Sex", "Age")
# bi.kdeplot("Age", hue="Survived")

# # Categorical vs categorical
print(bi.crosstab("species", "island"))
# bi.heatmap("Pclass", "Survived")

# # Multiple numerical columns
# bi.pairplot(columns=["Age", "Fare"])

# # Grouped analysis
# print(bi.group_summary("Sex", "Age"))