# Association Rule Mining on Transaction Data

In this project, I delved into transaction data to discover product associations and relationships. I utilized the principles of association rule mining, specifically the Apriori algorithm, to reveal which products are frequently bought together. Additionally, I visualized the results with a colorful network graph to showcase these associations visually.

## Data Used

- **Transaction Data**: Contains lists of items for each transaction. Each list is a record of items purchased together in a single shopping session.

## Key Steps

1. **Data Loading**: Loaded the transaction data to kick things off.
2. **Data Preparation**: Processed and tidied up the data to make it fit for mining.
3. **Association Rule Mining**:
    - Used the Apriori algorithm to mine for product association rules.
    - Generated a set of rules showing products that tend to be bought together.
4. **Visualization**:
    - Created a vibrant network graph to depict product relationships visually.

## Packages Used

- `pandas`: For data manipulation and analysis.
- `mlxtend`: To perform the Apriori algorithm and generate association rules.
- `networkx` & `matplotlib`: For visualizing product relationships through a network graph.

