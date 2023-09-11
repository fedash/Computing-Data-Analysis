
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

retail_data = pd.read_excel('Online Retail.xlsx', engine='openpyxl')
retail_data.head()

#Step 1: Load data. Check for missing values
print("Number of rows:", retail_data.shape[0])
print("Number of columns:", retail_data.shape[1])
print("\n")

missing_values = retail_data.isnull().sum()
print("Missing values for each column:")
print(missing_values)
print("\n")

print("Descriptive statistics:")
print(retail_data.describe())

# ## Step 2: Data Preprocessing
retail_data = retail_data.dropna(subset=['Description'])

retail_data = retail_data[retail_data['Quantity'] > 0]
retail_data = retail_data[retail_data['UnitPrice'] > 0]

transactions = retail_data.groupby('InvoiceNo')['Description'].apply(list)
print(transactions.head())

# Step 3: Association Rule Mining
encoder = TransactionEncoder()
onehot = encoder.fit_transform(transactions)
onehot_df = pd.DataFrame(onehot, columns=encoder.columns_)

print("Shape:")
print(onehot_df.shape)
print("\n")

print("Top 10 items by frequency:")
product_frequencies = onehot_df.sum().sort_values(ascending=False)
print(product_frequencies.head(10))
print("\n")

print("Descriptive statistics:")
transaction_counts = onehot_df.sum(axis=1)
print(transaction_counts.describe())

frequent_itemsets = apriori(onehot_df, min_support = 0.025, use_colnames=True)
frequent_itemsets.tail()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules_sorted = rules.sort_values(by="lift", ascending=False)
rules_sorted.head(10)

# Step 4: Visualizing Association Rules
def normalize(values, width_range=(1, 4)):
    min_val, max_val = min(values), max(values)
    min_width, max_width = width_range
    return [min_width + (max_width - min_width) * (value - min_val) / (max_val - min_val) for value in values]

edge_widths = [row['lift'] for _, row in rules.iterrows()]
normalized_edge_widths = normalize(edge_widths)


graph = nx.DiGraph()
item_to_id = {}
id_to_item = {}

current_id = 0
for _, row in rules.iterrows():
    for item in row['antecedents']:
        if item not in item_to_id:
            item_to_id[item] = current_id
            id_to_item[current_id] = item
            current_id += 1
    for item in row['consequents']:
        if item not in item_to_id:
            item_to_id[item] = current_id
            id_to_item[current_id] = item
            current_id += 1

edge_widths = []
for _, row in rules.iterrows():
    graph.add_edge(item_to_id[tuple(row['antecedents'])[0]], item_to_id[tuple(row['consequents'])[0]], weight=row['lift'])
    edge_widths.append(row['lift'])

colors = plt.cm.rainbow(np.linspace(0, 1, len(list(nx.weakly_connected_components(graph)))))
color_map = {}
for idx, component in enumerate(nx.weakly_connected_components(graph)):
    for node in component:
        color_map[node] = colors[idx]

node_sizes = [len(list(graph.neighbors(n))) * 150 for n in graph.nodes()]

fig, ax = plt.subplots(figsize=(16, 8))
pos = nx.spring_layout(graph, k=0.6)
nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_sizes, node_color=[color_map[node] for node in graph.nodes()])
nx.draw_networkx_labels(graph, pos, ax=ax, labels={node:node for node in graph.nodes()}, font_size=10)
nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='gray', width=normalized_edge_widths, alpha=0.6, arrows=False)

ax.set_title('Network Graph of Association Rules')

handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{id}: {item}", markersize=8, markerfacecolor=color_map[id]) for id, item in id_to_item.items()]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), title="Item ID")

plt.tight_layout()
plt.show()
