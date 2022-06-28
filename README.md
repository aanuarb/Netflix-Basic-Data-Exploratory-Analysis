# Netflix Exploratory Data Analysis V1

## About Dataset

This data set was created to list all shows available on Netflix streaming, and analyze the data to find interesting facts. This data was acquired in May 2022 containing data available in the United States.

## Step 1: Import the necessary libraries

import pandas as pd
from pandas import DataFrame, read_excel, merge
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import random
import plotly.graph_objects as go

## Step 2: Load dataset

data_titles = pd.read_csv(r'C:\Users\melli\OneDrive\Desktop\Python Portfolio\1. Netflix Analysis\Dataset\titles.csv')

## Step 3: Understand the dataset

data_titles.head()

data_titles.tail()

data_titles.shape

data_titles.nunique()

data_titles.describe()

## Step 4: Drop any columns that are unnecessary for this analysis

data_titles=data_titles.drop(['id', 'age_certification', 'runtime', 'seasons', 'imdb_id', 'imdb_votes', 'tmdb_popularity'], axis=1)

data_titles.head()

## Step 5: Clean dataset

data_titles.isnull().sum()

### Since we require imdb and tmdb score for this analysis, I used the mean value to replace the null value in the dataset

avg_imdb_score = data_titles['imdb_score'].astype('float').mean(axis=0)
data_titles['imdb_score'].replace(np.nan, avg_imdb_score,  inplace=True)

avg_tmdb_score = data_titles['tmdb_score'].astype('float').mean(axis=0)
data_titles['tmdb_score'].replace(np.nan, avg_tmdb_score,  inplace=True)

data_titles.isnull().sum()

### There are multiple variables within the genres value. To avoid multiple duplicates within the column, I've used the lambda function to only select the first variable in each value.

data_titles['genres'] = data_titles['genres'].apply(lambda x: x.split(',')[0])

data_titles.head()

## Step 6: Perform exploratory data analysis

### Which type of content is more widely available on Netflix

data_titles.groupby(['type']).sum().plot(kind='pie', y='imdb_score', title='Type of Content on Netflix', colors=['red', 'grey'], autopct='%1.0f%%')

### What type of genres are more widely available on Netflix?

sns.countplot(data_titles['genres'], palette='winter')

plt.title('Number of genres on Netflix')
plt.xticks(rotation='vertical', size=8)
plt.xlabel('genres')
plt.ylabel('count')
plt.show()

### Which genres garners a high imdb score?

comedy = data_titles[data_titles['genres'] == "['comedy']"]
drama = data_titles[data_titles['genres'] == "['drama']"]
thriller = data_titles[data_titles['genres'] == "['thriller']"]
romance = data_titles[data_titles['genres'] == "['romance']"]
action = data_titles[data_titles['genres'] == "['action']"]
crime = data_titles[data_titles['genres'] == "['crime']"]
documentation = data_titles[data_titles['genres'] == "['documentation']"]
scifi = data_titles[data_titles['genres'] == "['scifi']"]
animation = data_titles[data_titles['genres'] == "['animation']"]
reality = data_titles[data_titles['genres'] == "['reality']"]
horror = data_titles[data_titles['genres'] == "['horror']"]
family = data_titles[data_titles['genres'] == "['family']"]
music = data_titles[data_titles['genres'] == "['music']"]
war = data_titles[data_titles['genres'] == "['war']"]
western = data_titles[data_titles['genres'] == "['western']"]
fantasy = data_titles[data_titles['genres'] == "['fantasy']"]
history = data_titles[data_titles['genres'] == "['history']"]
sport = data_titles[data_titles['genres'] == "['sport']"]

mean_values = [comedy['imdb_score'].mean(), drama['imdb_score'].mean(), thriller['imdb_score'].mean(), romance['imdb_score'].mean(),
              action['imdb_score'].mean(), crime['imdb_score'].mean(), documentation['imdb_score'].mean(), 
               scifi['imdb_score'].mean(), animation['imdb_score'].mean(), reality['imdb_score'].mean(),
              horror['imdb_score'].mean(), family['imdb_score'].mean(), music['imdb_score'].mean(),
              war['imdb_score'].mean(), western['imdb_score'].mean(), fantasy['imdb_score'].mean(), 
               history['imdb_score'].mean(), sport['imdb_score'].mean()]

genres = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

imdb_plot = sns.barplot(x = genres, y = mean_values)
imdb_plot.set_xlabel('Genres')
imdb_plot.set_ylabel('Average tmdb Score')
imdb_plot.set_xticklabels(['Comedy', 'Drama', 'Thriller', 'Romance', 'Action', 'Crime', 'Documentation', 'Scifi', 'Animation', 'Reality',
                          'Horror', 'Family', 'Music', 'War', 'Western', 'Fantasy', 'History', 'Sport'])  
plt.xticks(rotation='vertical', size=10)
plt.show()

### Which movies have the highest imdb score? 

# Top 10 movies with highest rating based on imdb score
top_20_imdb_score = data_titles.sort_values(['imdb_score'], ascending= False)[['title','imdb_score']].head(20)

top_20_imdb_score.plot(kind='barh', x = 'title', y = 'imdb_score', figsize=(9, 6), color = 'brown')
plt.title('Top 20 Movie/Shows Based on IMDB Scores')
plt.xlabel('IMDB Score')
plt.ylabel('Title')
plt.show()

### Which actor/director are most popular on Netflix based on TMDB Popularity? 

# Top 10 names with highest rating based on tmdb score and popularity
top_20_tmdb_rating = data_id.sort_values(['tmdb_score', 'tmdb_popularity'], ascending= False)[['name','tmdb_score','tmdb_popularity', 'type']].head(20)

top_20_tmdb_rating.plot(kind='barh', x = 'name', y = 'tmdb_popularity', figsize=(9, 6), color = 'brown')
plt.title('Top 20 Actors/Directors based on tmdb popularity')
plt.xlabel('tmdb_popularity')
plt.ylabel('Actor/Director')
plt.show()

US = data_id[data_id['production_countries']== 'US']
top_20_show = US.sort_values(['title', 'imdb_score'], ascending= False)[['title','imdb_score','production_countries', 'type']].head(20)



## Step 7: Create basic recommendation tool

### Merged credits dataset with title dataset to include actors name in the recommendation

data_credits = pd.read_csv(r'C:\Users\melli\OneDrive\Desktop\Python Portfolio\1. Netflix Analysis\Dataset\credits.csv')
data_source_titles=pd.read_csv(r'C:\Users\melli\OneDrive\Desktop\Python Portfolio\1. Netflix Analysis\Dataset\titles.csv')

data_id = data_credits.merge(data_source_titles, on='id', how='left')
data_id.head(1)

data_recommendation=data_id.copy()

data_recommendation=data_recommendation.drop(['person_id','id','role','type', 
                                              'release_year','seasons','production_countries','imdb_votes',
                                              'imdb_id','tmdb_popularity'], axis=1)

n_samples = 10

for _ in range(n_samples):
    i = random.choice(range(data_recommendation.shape[0]))
    print(f"REVIEW TEXT\n\nTitle: {data_recommendation['title'][i]}\nActor: {data_recommendation['name'][i]}\nGenres: {data_recommendation['genres'][i]}\nGenres: {data_recommendation['genres'][i]}\nRuntime:{data_recommendation['runtime'][i]} \n\nIMDB:{data_recommendation['imdb_score'][i]}\nRotten Tomato:{data_recommendation['tmdb_score'][i]}")
    print('\n', 90*"-", '\n')

# END
