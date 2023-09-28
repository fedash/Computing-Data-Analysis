import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from scipy import stats

df_customers = pd.read_csv('data/olist_customers_dataset.csv')
df_items = pd.read_csv('data/olist_order_items_dataset.csv')
df_payment = pd.read_csv('data/olist_order_payments_dataset.csv')
df_orders = pd.read_csv('data/olist_orders_dataset.csv')
df_products = pd.read_csv('data/olist_products_dataset.csv')
df_sellers = pd.read_csv('data/olist_sellers_dataset.csv')
df_translation = pd.read_csv('data/product_category_name_translation.csv')

df_products = df_products.merge(df_translation, on='product_category_name', how='left')
df_products.drop(columns=['product_category_name'], inplace=True)
df_products.rename(columns={'product_category_name_english': 'product_category_name'}, inplace=True)

dataframes = {
    'df_customers': df_customers, 
    'df_items': df_items, 
    'df_payment': df_payment, 
    'df_orders': df_orders, 
    'df_products': df_products, 
    'df_sellers': df_sellers
}

for name, df in dataframes.items():
    print(f'{name}:\n')
    print(df.info())
    print(df.head(3))
    print(df.describe())
    print("\n----------------------------------------\n")

df_products.rename(columns={'product_name_lenght': 'product_name_length', 
                            'product_description_lenght': 'product_description_length'}, inplace=True)

df_orders.dropna(subset=['order_approved_at'], inplace=True)
df_orders['order_delivered_carrier_date'] = pd.to_datetime(df_orders['order_delivered_carrier_date'])
df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
avg_delivery_to_carrier_time = (df_orders['order_delivered_carrier_date'] - df_orders['order_purchase_timestamp']).mean()
df_orders['order_delivered_carrier_date'].fillna(df_orders['order_purchase_timestamp'] + avg_delivery_to_carrier_time, inplace=True)
df_orders['order_delivered_customer_date'].fillna('Not Delivered Yet', inplace=True)
df_products['product_name_length'].fillna(df_products['product_name_length'].mean(), inplace=True)
df_products['product_description_length'].fillna(df_products['product_description_length'].mean(), inplace=True)
df_products['product_photos_qty'].fillna(df_products['product_photos_qty'].mean(), inplace=True)
df_products['product_weight_g'].fillna(df_products['product_weight_g'].mean(), inplace=True)
df_products['product_length_cm'].fillna(df_products['product_length_cm'].mean(), inplace=True)
df_products['product_height_cm'].fillna(df_products['product_height_cm'].mean(), inplace=True)
df_products['product_width_cm'].fillna(df_products['product_width_cm'].mean(), inplace=True)
df_products.dropna(subset=['product_category_name'], inplace=True)

time_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date','order_estimated_delivery_date']
for col in time_cols:
    df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')
df_items['shipping_limit_date'] = pd.to_datetime(df_items['shipping_limit_date'], errors='coerce')

df_merged = pd.merge(df_orders, df_customers, on='customer_id', how='left')
df_merged = pd.merge(df_merged, df_items, on='order_id', how='left')
df_merged = pd.merge(df_merged, df_products, on='product_id', how='left')
df_merged = pd.merge(df_merged, df_payment, on='order_id', how='left')
df_merged = pd.merge(df_merged, df_sellers, on='seller_id', how='left')

df_merged.drop(columns=['customer_id', 'order_id', 'product_id', 'seller_id'], inplace=True)
df_merged.head()

df_merged['sales_value'] = df_merged['order_item_id'] * df_merged['price']

def classify_spending(x):
    if x < 100:
        return 'low'
    elif x < 170:
        return 'medium'
    else:
        return 'high'

df_merged['spending_category'] = df_merged['payment_value'].apply(classify_spending)

df_merged['customer_state'] = df_merged['customer_state'].astype('category')
df_merged['product_category_name'] = df_merged['product_category_name'].astype('category')

df_merged['volume_cm3'] = df_merged['product_length_cm'] * df_merged['product_height_cm'] * df_merged['product_width_cm']
df_merged.drop(columns=['product_length_cm', 'product_height_cm', 'product_width_cm'], inplace=True)

df_merged['order_purchase_month'] = pd.to_datetime(df_merged['order_purchase_timestamp']).dt.month
df_merged.head()

popular_products = df_merged.groupby('product_category_name')['order_item_id'].sum().sort_values(ascending=False)
print(popular_products.head())

plt.figure(figsize=(10, 6))
popular_products.head(10).plot(kind='bar')
plt.title('Top 10 Most Popular Product Categories')
plt.ylabel('Quantity Sold')
plt.show()

preferred_payment_methods = df_merged.groupby('payment_type').size()
total = preferred_payment_methods.sum()
preferred_payment_methods_pct = (preferred_payment_methods / total) * 100
print(preferred_payment_methods_pct)

preferred_payment_methods_pct.plot(kind='bar', figsize=(8, 5))
plt.title('Preferred Payment Methods')
plt.ylabel('Percentage of Orders (%)')
plt.xlabel('Payment Types')
plt.xticks(rotation=45)
plt.show()

order_flows = df_merged.groupby(['customer_state', 'seller_state']).size().unstack().fillna(0)
total = order_flows.sum().sum()
order_flows_pct = (order_flows / total) * 100

plt.figure(figsize=(7, 7))
sns.heatmap(order_flows_pct, cmap='Blues', annot=False)  
plt.title('Order Flows Across Different States (%)')
plt.show()

df_merged['order_purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'])
df_merged['order_delivered_customer_date'] = pd.to_datetime(df_merged['order_delivered_customer_date'])

df_merged['delivery_time'] = (df_merged['order_delivered_customer_date'] - df_merged['order_purchase_timestamp']).dt.days

avg_delivery_time_by_category = df_merged.groupby('product_category_name')['delivery_time'].mean()
avg_delivery_time_by_state = df_merged.groupby('customer_state')['delivery_time'].mean()

print(avg_delivery_time_by_category.head())
print(avg_delivery_time_by_state.head())

avg_delivery_time_by_category.sort_values().plot(kind='bar', figsize=(12, 6))
plt.title('Average Delivery Time by Product Category')
plt.ylabel('Average Delivery Time (days)')
plt.show()

avg_delivery_time_by_state.sort_values().plot(kind='bar', figsize=(12, 6))
plt.title('Average Delivery Time by State')
plt.ylabel('Average Delivery Time (days)')
plt.show()

correlation_data = df_merged[['volume_cm3', 'product_weight_g', 'delivery_time']]
delivery_corr = correlation_data.corr()

plt.figure(figsize=(5,5))
sns.heatmap(delivery_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Delivery Time, Product Volume, and Weight')
plt.show()

correlation_description_sales = df_merged[['order_item_id', 'product_description_length']].corr()
plt.figure(figsize=(7, 6))
plt.hexbin(data=df_merged, x='product_description_length', y='order_item_id', C=df_merged['order_item_id'], 
           gridsize=20, cmap='Blues', reduce_C_function=np.mean)
cb = plt.colorbar(label='Average in Bin')
plt.title('Hexbin Plot of Product Description Length and Average Item Quantity Sold')
plt.xlabel('Product Description Length')
plt.ylabel('Average Item Quantity Sold')
plt.show()

average_freight_by_category = df_merged.groupby('product_category_name')['freight_value'].mean().reset_index()
average_freight_by_category = average_freight_by_category.sort_values(by='freight_value', ascending=False)
plt.figure(figsize=(12, 14))
sns.barplot(data=average_freight_by_category, 
            x='freight_value', 
            y='product_category_name', 
            orient='h', 
            order=average_freight_by_category['product_category_name'])
plt.title('Average Freight Value by Product Category')
plt.xlabel('Average Freight Value')
plt.ylabel('Product Category')
plt.show()


corr = df_merged[['price', 'freight_value', 'product_name_length', 'product_description_length', 'product_weight_g', 'payment_value', 'volume_cm3', 'delivery_time']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

monthly_sales = df_merged.groupby(['order_purchase_month', 'spending_category'])['sales_value'].sum().unstack().reset_index()

monthly_sales.plot(kind='bar', stacked=True, x='order_purchase_month', figsize=(12,6))
plt.title('Monthly Sales by Spending Category')
plt.xlabel('Order Purchase Month')
plt.ylabel('Total Sales Value')
plt.show()

df_merged['delivery_time'] = (pd.to_datetime(df_merged['order_delivered_customer_date']) - pd.to_datetime(df_merged['order_estimated_delivery_date'])).dt.days
df_merged['total_delivery_time'] = (pd.to_datetime(df_merged['order_delivered_customer_date']) - pd.to_datetime(df_merged['order_purchase_timestamp'])).dt.days
total_sales_value = df_merged.groupby('customer_state')['sales_value'].sum().reset_index()
average_freight_value = df_merged.groupby('customer_state')['freight_value'].mean().reset_index()
average_total_delivery_time = df_merged.groupby('customer_state')['total_delivery_time'].mean().reset_index()

geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

fig1 = px.choropleth(average_total_delivery_time, 
                     geojson=geojson_url, 
                     locations='customer_state', 
                     featureidkey="properties.sigla",
                     color='total_delivery_time',
                     title="Average Total Delivery Time by State in Brazil")
fig1.update_geos(fitbounds="locations")
fig1.show()

from PIL import Image
im = Image.open("DeliveryTimeMap.png")
im.show()

fig2 = px.choropleth(total_sales_value, 
                     geojson=geojson_url, 
                     locations='customer_state', 
                     featureidkey="properties.sigla",
                     color='sales_value',
                     title="Total Sales Value by State in Brazil")
fig2.update_geos(fitbounds="locations")
fig2.show()

im = Image.open("OrdersMap.png")
im.show()

fig3 = px.choropleth(average_freight_value, 
                     geojson=geojson_url, 
                     locations='customer_state', 
                     featureidkey="properties.sigla",
                     color='freight_value',
                     title="Average Freight Value by State in Brazil")
fig3.update_geos(fitbounds="locations")
fig3.show()

im = Image.open("FreightValueMap.png")
im.show()