from logging import NullHandler
from unicodedata import numeric
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_pandas

#Enginering Features from Raw Data
def average_stats(df, index, seen_player, row):
    for player_index, value in df.loc[index].items():
        if isinstance(value, float):
            df.loc[index] = df.loc[index].replace(to_replace=df.loc[index][player_index], value=seen_player[row['Player']][0][player_index] / seen_player[row['Player']][1])

playerDF= pd.DataFrame(columns=['Player', 'Match Up', 'Game Date', 'W/L', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-'])
astypeDict= {'Player':str, 'Match Up':str, 'Game Date':str, 'MIN':float, 'PTS':float, 'FG%':float, '3P%':float, 'FT%':float, 'OREB':float, 'DREB':float, 'AST':float, 'STL':float, 'BLK':float, 'TOV':float, 'PF':float, '+/-':float}
groupingDict= {'Player':'first', 'Match Up':'first', 'Game Date':'first', 'MIN':'sum', 'PTS':'sum', 'FG%':'sum', '3P%':'sum', 'FT%':'sum', 'OREB':'sum', 'DREB':'sum', 'AST':'sum', 'STL':'sum', 'BLK':'sum', 'TOV':'sum', 'PF':'sum', '+/-':'sum'}
frames=[]
for year in tqdm(range(8, 9)):
    print(year)
    df = pd.read_csv('/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/PlayerSchedule'+str(year//10) + str(year%10) + '-'+ str((year+1)//10) + str((year+1)%10) +'.csv')
    df=df.drop(columns=["Season", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "REB"])
    ftMean = np.mean([float(val) for val in df["FT%"] if val!='-'])
    p3Mean = np.mean([float(val) for val in df["3P%"] if val!='-'])
    fgMean = np.mean([float(val) for val in df["FG%"] if val!='-'])
    df["FT%"].replace('-', str(ftMean), inplace=True)
    df["3P%"].replace('-', str(p3Mean), inplace=True)
    df["FG%"].replace('-', str(fgMean), inplace=True)
    df=df.astype(astypeDict)
    # sort by games from earliest to latest game
    df.sort_values(by='Game Date')
    df = df.iloc[::-1]
    seen_player = {}
    count = 0
    for index, row in tqdm(df.iterrows()):
        if row['Player'] not in seen_player:
            seen_player[row['Player']] = row, 1
            continue
        else:
            prev_info = seen_player[row['Player']][0]
            prev_info.loc['Team'] = row['Team']
            prev_info.loc['Match Up'] = row['Match Up']
            prev_info.loc['Game Date'] = row['Game Date']
            prev_info.loc['W/L'] = row['W/L']
            prev_info.loc[['MIN', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']] += row[['MIN', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']]
            seen_player[row['Player']] = prev_info, seen_player[row['Player']][1] + 1
        average_stats(df, index, seen_player, row)
    frames.append(df)

playerDF = pd.concat(frames)
print(type(playerDF))
print(playerDF)
pd.DataFrame(playerDF).to_csv("/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/NBA_Average_Stats.csv", header=['Player', 'Match Up', 'Game Date', 'W/L', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-'], index=None)
        

    

    