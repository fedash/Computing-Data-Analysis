import requests
import pandas as pd
import time
from collections import Counter
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

api_token = input("Please enter your GitHub API Token: ")

headers = {
    "Authorization": f"token {api_token}",
    "Accept": "application/vnd.github.v3+json"
}

base_url = "https://api.github.com/search/repositories?q=stars:>1&sort=stars&order=desc&per_page=100&page="

all_repos = []

for page_num in range(1, 11):
    print(f"Fetching data for page {page_num}...")
    
    response = requests.get(base_url + str(page_num), headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch data for page {page_num}. Status code: {response.status_code}")
        break
    
    if 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) <= 1:
        reset_time = int(response.headers['X-RateLimit-Reset'])
        sleep_duration = reset_time - time.time() + 5
        print(f"Rate limit exceeded. Sleeping for {sleep_duration} seconds.")
        time.sleep(sleep_duration)
    else:
        data_page = response.json()
        all_repos.extend(data_page['items'])

print(f"Fetched data for {len(all_repos)} repositories.")

print("\nExample - one of the repositories:")
print(all_repos[0])

df_repos = pd.DataFrame(all_repos)
print(f'\nThere are {len(df_repos)} repositories in our dataset.')

print(df_repos.head(2))

print(df_repos.describe())

plt.figure(figsize=(15, 8))
sns.countplot(data=df_repos, y='language', order=df_repos['language'].value_counts().index[:10], palette='viridis')
plt.title('Top 10 Programming Languages in top-1000 Repositories')
plt.xlabel('Count')
plt.ylabel('Programming Language')
plt.show()

plt.figure(figsize=(15, 8))
sns.histplot(df_repos['stargazers_count'], kde=True, bins=50)
plt.title('Distribution of Stars in Repositories')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Repositories')
plt.show()

plt.figure(figsize=(15, 8))
sns.histplot(df_repos['forks_count'], kde=True, bins=50)
plt.title('Distribution of Forks in Repositories')
plt.xlabel('Number of Forks')
plt.ylabel('Number of Repositories')
plt.show()

all_topics = [topic for sublist in df_repos['topics'].tolist() for topic in sublist]

topic_counts = Counter(all_topics)
top_10_topics = topic_counts.most_common(10)
print(top_10_topics)

topic_counts_all = dict(topic_counts)

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno', max_words=100).generate_from_frequencies(topic_counts_all)

plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Topics Word Cloud Based on Occurrence Frequency')
plt.show()

avg_stars_all = {}
all_unique_topics = set(all_topics)

for topic in all_unique_topics:
    avg_stars_all[topic] = df_repos[df_repos['topics'].apply(lambda x: topic in x)]['stargazers_count'].mean()

sorted_avg_stars = dict(sorted(avg_stars_all.items(), key=lambda item: item[1], reverse=True)[:10])

topics_sorted = list(sorted_avg_stars.keys())
stars_sorted = list(sorted_avg_stars.values())
print(topics_sorted)

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno', max_words=100).generate_from_frequencies(avg_stars_all)

plt.figure(figsize=(12, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Topics Word Cloud Based on Average Star Count')
plt.show()