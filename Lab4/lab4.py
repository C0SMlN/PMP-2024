import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {'Category': ['A', 'A', 'A', 'A', 'A', 'A'],
        'Value': [10, 20, 15, 25, 15, 34]}

# Creating a DataFrame
df = pd.DataFrame(data)

# Creating the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Category', y='Value', data=df)
plt.title('Boxplot of Values by Category')
plt.show()
