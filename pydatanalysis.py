import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from numpy import log1p

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv(r"uncleaned zomato dataset.csv")

print(df.head())
df.info()
print(df.isnull().sum())

# Convert cost column to numeric
df['cost_for_two'] = pd.to_numeric(df['cost_for_two'], errors='coerce')

# Drop rows with missing essential values
df.dropna(subset=['rating', 'rating_count', 'cuisine', 'cost_for_two'], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Filter: Top 10 cities by restaurant count
top_cities = df['city'].value_counts().nlargest(10).index.tolist()
df = df[df['city'].isin(top_cities)]

### Step 3: Feature Engineering

# Price buckets
bins = [0, 300, 600, 1000, df["cost_for_two"].max()]
labels = ['Low', 'Mid', 'High', 'Premium']
df['price_bucket'] = pd.cut(df['cost_for_two'], bins=bins, labels=labels)

# Explode cuisines
df['cuisine'] = df['cuisine'].str.split(', ')
df_exploded = df.explode('cuisine')

### Step 4: Univariate and Bivariate EDA
# counts, bins = np.histogram(df['rating'], bins=20)
# print(bins)
# print(df['rating'].min())

# Distribution of ratings
plt.figure(figsize=(8, 5))
sns.histplot(df['rating'], bins=20, color="#4682B4")
plt.title('Distribution of Restaurant Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Cost vs Rating scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(x='cost_for_two', y='rating', data=df, alpha=0.6)
plt.title('Cost for Two vs Rating')
plt.xlabel('Cost for Two (INR)')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

### Step 5: Insight 1 – Price–Rating Elasticity
grouped = df.groupby(['city', 'price_bucket'])['rating'].mean().unstack()
grouped['Elasticity_Score'] = (grouped['Premium'] - grouped['Low']) / 3

# Plot elasticity score
plt.figure(figsize=(10, 6))
grouped['Elasticity_Score'].sort_values().plot(kind='barh', color='orange')
plt.title('Rating Elasticity by City')
plt.xlabel('Elasticity Score')
plt.ylabel('City')
plt.tight_layout()
plt.show()

### Step 6: Insight 2 – Cuisine Sentiment Index (CSI)

# Group by city and cuisine, count how many restaurants each pair has
cuisine_group = df_exploded.groupby(['city', 'cuisine']).agg({
    'rating': ['mean', 'std', 'count'],
    'rating_count': 'mean'
}).reset_index()

cuisine_group.columns = ['City', 'Cuisine', 'Avg_Rating', 'Std_Dev', 'Count', 'Avg_Votes']

cuisine_group = cuisine_group[cuisine_group['Count'] >= 3]
cuisine_group['Std_Dev'].fillna(0, inplace=True)
cuisine_group['CSI'] = ( cuisine_group['Avg_Rating'] * log1p(cuisine_group['Avg_Votes'])/ (cuisine_group['Std_Dev'] + 0.1))

# Sort and show top 10 cuisines
top_csi = cuisine_group.sort_values(by='CSI', ascending=False).head(10)
print("Top 10 cuisines by CSI (Cuisine Sentiment Index):")
print(top_csi)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=top_csi, x='CSI', y='Cuisine', hue='City')
plt.title('Top 10 Cuisines by Cuisine Sentiment Index')
plt.xlabel('CSI Score')
plt.ylabel('Cuisine')
plt.tight_layout()
plt.show()

### Step 7: Insight 3 – Saturation vs. Rating Decay
# Extract locality
df['Locality'] = df['address'].str.extract(r'(.+?),')
locality_group = df.groupby('Locality').agg({
    'name': 'count',
    'rating': 'mean'
}).rename(columns={'name': 'Restaurant_Count'})
locality_group = locality_group[locality_group['Restaurant_Count'] >= 10]
print(locality_group)

# Correlation
print("Correlation between restaurant count and average rating:")
print(locality_group.corr())

# Visualize
plt.figure(figsize=(8, 5))
sns.regplot(data=locality_group, x='Restaurant_Count', y='rating', scatter_kws={'alpha':0.5})
plt.title('Restaurant Count vs Average Rating by Locality')
plt.xlabel('Number of Restaurants in Locality')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.show()

# Save cleaned and enriched dataset
# df.to_csv("cleaned zomato dataset.csv",index="false")