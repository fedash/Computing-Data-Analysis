# GitHub Repositories Analysis

This project focuses on analyzing the top repositories on GitHub. By leveraging the GitHub API, the project mines data on the repositories with the highest star counts to derive insights on popular languages, topics, and general engagement (stars and forks). 

## Overview

1. **Data Collection:** Used the GitHub API to fetch data for the top 1000 repositories based on star count.
2. **Data Analysis:** 
   - Identified the top 10 programming languages used across these repositories.
   - Analyzed the distribution of stars and forks across repositories.
   - Examined the trending topics associated with these repositories.
   - Visualized the topics in the form of word clouds, once based on occurrence frequency and then based on average star count.

## Main Steps:

1. **Data Extraction:** Extracted data from the GitHub API spanning across 10 pages, each containing information on 100 repositories.
2. **Data Exploration:** Explored the general statistics of the data and observed the structure of the repositories.
3. **Visual Analysis:** Plotted histograms and bar plots to understand the distributions and popular choices among repositories.
4. **Topic Analysis:** Extracted all topics across repositories and showcased them in a word cloud. Further dived into understanding which topics are not just popular but also highly starred on average.

## Packages Used:

- `requests`: For making API requests.
- `pandas`: For data manipulation and analysis.
- `time`: To handle the rate limits of the GitHub API.
- `collections`: Specifically, the `Counter` to count occurrences.
- `numpy`: For numerical operations.
- `wordcloud`: For generating word clouds.
- `matplotlib`: For visualization.
- `seaborn`: Advanced visualization.
