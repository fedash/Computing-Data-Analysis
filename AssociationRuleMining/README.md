# Association Rule Mining on Transaction Data


This project delves into transaction data to uncover product associations and relationships. Using the principles of association rule mining, specifically the Apriori algorithm, it reveals which products are frequently bought together. The findings are then visualized with a network graph to showcase these associations visually.

## Data Used

- **[Online Retail Data Set](https://archive.ics.uci.edu/ml/datasets/Online+Retail)**: The UCI Online Retail dataset contains transactional data showing sales for an online retail company based in the UK, primarily detailing products, quantities, and purchase timestamps.

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

