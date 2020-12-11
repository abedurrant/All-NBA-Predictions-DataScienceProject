# All-NBA Team Predictions
## Introduction

All-NBA teams are supposed to be awards given to the best players in basketball for any given season. These awards are subjective and are based upon people voting their opinions, but the people voting are basing their opinion off of both watching games and statistics. As a basketball fan I love paying attention to the NBA awards/All-NBA teams and I also like to speculate before the season about which players are going to do well. It is common among fans to make predictions before the season, so I wanted to see if I could create a model to better predict NBA awards before the season. The specific question I wanted to answer for this project was whether or not it is possible to predict who will make an all-NBA team for a given season before that season even starts? In other words, could I use data from the previous season a player played in to project whether or not they would be on an all-NBA team in the next season?

## Data Collection and Data Cleaning
I began my project by web-scraping player data from basketball-reference.com. I decided to take a wide variety of different statistics and so I included per-game averages, advanced statistics like Win Shares or True Shooting percentage, play-by-play data, shooting data, and per 100 possession data. All of these statistics were available on basketball-reference.com and I was able to scrape them using simple pandas read_html and a simple for loop. Here is an example:

```python
player_df = pd.DataFrame()
for i in range(season_min, season_max+1):
    link = "https://www.basketball-reference.com/leagues/NBA_" + str(i) + "_per_game.html"
    temp_df = pd.read_html(link)[0]
    temp_df['year'] = i
    player_df = pd.concat([player_df, temp_df])
```

I decided to base my analysis off of the previous 10 seasons worth of data, so 2010-2019. The NBA changes over time along with what people value most in a player, so using the previous 10 seasons seemed to give me enough data to build a powerful model while not going back too far to a different era of the NBA. I combined all the data I scraped from the different datasets on basketball reference into one overall player data set which I labeled player_all.

After getting the relevant player data from the past 10 seasons, I added to the dataset whether or not they made an all-NBA team to the dataset. I pulled this information from https://en.wikipedia.org/wiki/All-NBA_Team. It was simple enough to then just add another column to my overall data for whether a player made an all-NBA team in the next season by comparing my player_all dataframe with the new all-NBA data. I also added as a variable whether or not they had made the all-NBA that season as I felt that may be a good predictor of whether or not they would make the all-NBA in the next season. I then cleaned the data by removing repeating columns, removing extra header columns, changing datatypes, replacing missing values with 0 (as in all cases that I could see, the reason data was missing from basketball-reference was that the player had 0 of something such as minutes played at C), and filtering it to include only players that had played a minimum of 55 games. This data can be found in the NBAplayerData.csv file. It includes all seasons from 2010-2019. The player data for 2020 can be found in the NBAplayerData2020.csv file.

## EDA
The following is a table showing the percent increase in various statistics if a player is on an all-NBA team in the next season compared to a player not on an all-NBA team in the next season.

The following are charts looking at a few of the most used summary statistics of a player and how that compares from those that made all-NBA teams vs. those that didn't. Once again to clarify, this is if they made the all-NBA team in the following season as that is what we are focused on predicting.
<img src="https://github.com/abedurrant/All-NBA-Predictions-DataScienceProject/blob/main/All-NBA_Plot1.png" width="450" height="300">
<img src="https://github.com/abedurrant/All-NBA-Predictions-DataScienceProject/blob/main/All-NBA_Plot2.png" width="450" height="300">
<img src="https://github.com/abedurrant/All-NBA-Predictions-DataScienceProject/blob/main/All-NBA_Plot3.png" width="450" height="300">

## Method and Results
I decided to test a variety of Machine Learning classification methods to attempt to predict the all-NBA awards for the next season. I used a logistic regression model, a decision tree model, a random forest model, and a gradient boosting model. I picked these due to their generally good predictive performance and due to the fact that they were easy to implement with my model. I wanted to try a variety of different classification methods in order to see which methods performed best. The simple logistic regression model I used as sort of my baseline to compare other models too. The random forest model I decided on as it is generally seen as one of the stronger classification methods and can give insight into which features are important. However, it can potentially overfit a model so I attempted to optimize some of the parameters (this was actually difficult due to running my data through a pipeline, but I will discuss that in the next paragraph). I also decided to try a gradient boosting model due to its flexibility and reputation. However, like random forest it can have problems with overfitting or overemphasizing outliers. Finally, I tried a decision tree model as well to have a 4th option. 

I first ran the data through a pipeline that scaled that data and converted categorical columns so that they could be used. I attempted to add in polynomial features with interactions, but this negatively effected model performace, so I decided against leaving that in. I then separated the data into a train and test set in order to evaluate model performance later. I then fit my models to the data and generated both predictions and probabilities. I tried different parameters with the random forest model and gradient boosting model in order to try to find the optimal parameters. However, due to the complex nature of the pipeline as well as the large amount of categorical variables, it became hard to run functions to check the optimal parameters. However, after some trial and error I was able to find some parameters that worked well. I evaluated my models by looking at the accuracy, f1 score, precision score, recall score, and AUC score. The most important metrics in my mind were the precision score and recall score. These were both important as I mostly wanted my predictions for who would make an all-NBA team to be as accurate as possible. I also wanted to correctly predict as many of the players that did end up making the all-NBA team. For this reason both precision and recall were important as well as the overall accuracy score.

I adjusted my models by removing variables that seemed to have little effect on the outcome. I was suprised to find that removing the feature of whether or not the player made the all-NBA team in the previous season actually improved the model. I believe this could be to the models overfitting on that one variable as from the EDA I found that players who made the all-NBA in the previous season made the all-NBA the next season about 50% of the time. I also adjusted the threshold for some of the models and found that as I lowered it I was able to get a better recall score at the expense of precision. This made sense as by lower the threshold I was predicting more of the players who actually made the all-NBA team correct, but I was also predicting that more players would make it which lowered my precision from the ones that didn't. 

Overall the Random Forest and Gradient Boosting models performed the best. Logistic Regression tended to due quite good when given more variables, but as I discovered the variables that were most important, Random Forest and Gradient Boosting began to perform better. Logistic Regression also had a low recall score even though it had a high precision score. The random forest model tended to have a very good precision score, although its recall score was sometimes a bit lower than the Gradient Boosting model based on the training set. It had a precision score of .67 when I adjusted the threshold, which meant that of those it predicted to make the all-NBA team, 67% of them actually did. I consider this to be quite good as predicting NBA performance can be quite hard. It also came back with a recall score of .52, meaning that it correctly guessed 52% of those that actually did end up making the team. The other models all had precision scores in the lower 50's which made the Random Forest Model stand out even more. Due to this I decided to keep the Random Forest model to generate my predictions for 2020.

Overall I was happy with the results of my model. NBA performance is a hard thing to predict and different players seem to play on different levels from season to season, so the fact that my model was able to do as well as it did surprised me. There are definitely adjustments that can be made to improve it in the future, but I was able to generate a predictive model that estimated all-NBA teams with some accuracy which was my original research purpose and question.


## Conclusion

It is difficult to predict NBA awards. There are so many factors and things to consider that even though I used many statistics available, my model still was only able to predict about 50% of the actual all-NBA players. I was able to create a model to generate the predictions that I wanted and I found that a random forest model was the most optimal method to use. It tended to have overall accurate predictions, but estimating exactly who will make the teams was very tough. Each year in the NBA 15 players make the all-NBA teams, and I wasn't able to adjust my model to only take the top 15 predictions. Most of my models would predict less per year and so I believe that taking the top 15 players by the probability that the model generated will help me to predict a higher percentage of the players that will make it next season then the recall score indicates. In the future I want to add a few more features to my model that will possibly help it to generate better predictions than it did just using player performance metrics. I want to add in team performace in the form of wins as well as whether the player is in a contract year the next season. Both of these are things that people generally think affect whether or not a player is considered for an all-NBA team or plays well enough to make one. The model is limited due to the limited information it has on players (and can't account for players who missed a season due to injury), so adding some minor adjustments later on could improve it. I think removing players that were injured the following year could also help the model so that it doesn't take them into consideration. It is also potentially biased towards players that have stats very similar to those that make all-NBA teams a lot, althought most players who make all-NBA teams seem to have somewhat similar stats to previous all-NBA players.

This project also provided a template that I hope to use to predict all-NBA awards at the end of the season. I hope to be able to predict other awards as well by using similar data. 2021 will be a tough year to predict all-NBA teams for as Covid messed with the season last year and this year, but via my random forest model, here are the predictions for all-NBA in 2021:

| All-NBA First Team     | Position          | 
| ------------- |:-------------:| 
| Luka Dončić | G |
| James Harden    | G    |
| LeBron James | F      |
| Giannis Antetokounmpo | F |
| Anthony Davis   | C   |

| All-NBA Second Team     | Position          |
| ------------- |:-------------:|
| Damian Lillard | G |
| Ben Simmons  | G    |
| Kawhi Leonard | F      |
| Karl-Anthony Towns | F |
| Nikola Jokić  | C   |

| All-NBA Third Team     | Position          |
| ------------- |:-------------:|
| Russell Westbrook | G |
| Jimmy Butler   | G    |
| Jayson Tatum | F      |
| Bam Adebayo | F |
| Rudy Gobert   | C   |

Players that had high probabilities but aren't in the prediction due to position are Trae Young, Joel Embiid, Chris Paul and Devin Booker. If we count Jimmy Butler as a F then Bam Adebayo would not bee on th elist and would be replace by Trae Young at the guard spot.
