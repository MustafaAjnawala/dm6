# pip install pandas
# pip install mlxtend
# pip install numpy
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load data from CSV
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    print("Data Loaded Successfully!")
    print(df.head())  # Fixed: Add parentheses for head()
    return df

# Step 2: Apply Apriori algorithm to find frequent itemsets
def apply_apriori(df, min_support=0.5):
    df = df.drop(columns=['Transaction_ID'])  # Ensure 'Transaction_ID' exists in the dataset
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets

# Step 3: Generate association rules
def generate_association_rules(frequent_itemsets, min_confidence=0.7):
    # Assuming the total number of transactions is the sum of all item occurrences in 'support'
    num_itemsets = sum(frequent_itemsets['support'] * len(frequent_itemsets))  # Example calculation
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)
    return rules


def main():
    # Step 1: Load the dataset
    csv_file = "6_data.csv"  # Use consistent quotes
    data = load_data(csv_file)  # Pass the correct variable name
    frequent_itemsets = apply_apriori(data, min_support=0.5)
    print("\nFrequent Itemsets:\n", frequent_itemsets)
    rules = generate_association_rules(frequent_itemsets, min_confidence=0.7)
    print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

if __name__ == "__main__":
    main()
