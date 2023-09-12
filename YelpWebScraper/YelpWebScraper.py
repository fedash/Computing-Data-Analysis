import requests
import re
from bs4 import BeautifulSoup
import json
import pandas as pd
import html
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from geopy.geocoders import ArcGIS

# Step 1: Data Collection
base_url = "https://www.yelp.com/search?find_desc=Restaurants&find_loc=Erlangen%2C+BY&attrs=german&start={}"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

all_soups = []

for i in range(0, 80, 10):
    response = requests.get(base_url.format(i), headers=headers, timeout=100)
    if response.status_code != 200:
        raise Exception(f"Webpage fetch failed for page starting at {i}! Status code: {response.status_code}")
    soup = BeautifulSoup(response.content, 'html.parser')
    all_soups.append(soup)

# Step 2: Extracting Restaurant Details

restaurants = []

def extract_details(soup):
    script_content = None
    for script in soup.find_all("script"):
        if "@context" in script.text and "@type" in script.text and "ImageObject" in script.text:
            script_content = script.text
            break

    if not script_content:
        print("Script containing restaurant data not found!")
        return

    try:
        data = json.loads(script_content)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', script_content)
        if match:
            data_str = match.group(0)
            data = json.loads(data_str)
        else:
            print("Failed to parse JSON.")
            return

    for entry in data:
        if (
            entry.get("@type") == "ImageObject" and 
            entry.get("contentLocation") and 
            entry.get("contentLocation").get("address") and 
            entry.get("aggregateRating")
        ):
            address_info = entry["contentLocation"]["address"]
            restaurant_name = entry["aggregateRating"]["itemReviewed"]["name"]
            street_address = address_info["streetAddress"]
            review_count = entry["aggregateRating"]["reviewCount"]
            rating = entry["aggregateRating"]["ratingValue"]

            restaurants.append({
                'Name': restaurant_name,
                'Address': street_address,
                'Review Count': review_count,
                'Rating': rating
            })

for soup in all_soups:
    extract_details(soup)

for restaurant in restaurants[:3]:  
    print(f"Name: {restaurant['Name']}")
    print(f"Address: {restaurant['Address']}")
    print(f"Review Count: {restaurant['Review Count']}")
    print(f"Rating: {restaurant['Rating']}")
    print('-' * 50)

print(f"We have details on {len(restaurants)} German cuisine restaurants in Erlangen.")

# Step 3: Analyzing and Visualizing the Restaurant Data

for restaurant in restaurants:
    restaurant['Name'] = html.unescape(restaurant['Name'])

df_restaurants = pd.DataFrame(restaurants)
df_restaurants['Rating'] = df_restaurants['Rating'].round(2)
df_restaurants.head()

C = df_restaurants['Rating'].mean()
m = df_restaurants['Review Count'].quantile(0.25)

qualified = df_restaurants.copy().loc[df_restaurants['Review Count'] >= m]
qualified['Weighted Rating'] = round((qualified['Review Count'] / (qualified['Review Count'] + m) * qualified['Rating']) + (m / (qualified['Review Count'] + m) * C),2)

qualified = qualified.sort_values('Weighted Rating', ascending=False)
qualified.head(10)


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(df_restaurants['Rating'], bins=10, kde=True)
plt.title('Distribution of Restaurant Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Restaurants')
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(qualified['Weighted Rating'], bins=10, kde=True)
plt.title('Distribution of Restaurant Weighted Ratings')
plt.xlabel('Weighted Rating')
plt.ylabel('Number of Restaurants')
plt.show()

# Step 4. Top 10 German Cuisine Restaurants in Erlangen on the Map

top_10 = qualified[:10]

geolocator = ArcGIS(user_agent="geoapiExercises")

def get_lat_lon(address):
    location = geolocator.geocode(address + ", Erlangen, Germany")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

top_10_copy = top_10.copy()
latitudes, longitudes = zip(*top_10_copy['Address'].apply(get_lat_lon))
top_10_copy['Latitude'] = latitudes
top_10_copy['Longitude'] = longitudes
top_10_copy

m = folium.Map(location=[49.5971, 11.0063], zoom_start=13)

for index, row in top_10_copy.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Name'],
        icon=folium.Icon(icon="cutlery", prefix='fa')
    ).add_to(m)

m

m.save("erlangen_restaurants_map.html")


