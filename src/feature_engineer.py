from logging import NullHandler
from unicodedata import numeric
from venv import create
import pandas as pd
import numpy as np
from tqdm import tqdm, tqdm_pandas

#Enginering Features from Raw Data
def average_stats(df, index, seen_player, row):
    for player_index, value in df.loc[index].items():
        if isinstance(value, float):
            df.at[index, player_index] = seen_player[row['Player']][0][player_index] / seen_player[row['Player']][1]

def collect_all_traditional_stats():
    playerDF= pd.DataFrame(columns=['Player', 'Team', 'Match Up', 'Game Date', 'W/L', 'MIN', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-'])
    astypeDict= {'Player':str, 'Team':str, 'Match Up':str, 'Game Date':str, 'W/L':str, 'MIN':float, 'PTS':float, 'FG%':float, '3P%':float, 'FT%':float, 'OREB':float, 'DREB':float, 'REB':float, 'AST':float, 'STL':float, 'BLK':float, 'TOV':float, 'PF':float, '+/-':float}
    # groupingDict= {'Player':'first', 'Team': 'first', 'Match Up':'first', 'Game Date':'first', 'W/L':'first', 'MIN':'sum', 'PTS':'sum', 'FG%':'sum', '3P%':'sum', 'FT%':'sum', 'OREB':'sum', 'DREB':'sum', 'AST':'sum', 'STL':'sum', 'BLK':'sum', 'TOV':'sum', 'PF':'sum', '+/-':'sum'}
    frames=[]
    for year in tqdm(range(8, 23)):
        print(year)
        df = pd.read_csv('/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/PlayerSchedule'+str(year//10) + str(year%10) + '-'+ str((year+1)//10) + str((year+1)%10) +'.csv')
        if "Season" in df:
            df=df.drop(columns=["Season", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA"])
        else:
            df=df.drop(columns=["FGM", "FGA", "3PM", "3PA", "FTM", "FTA"])
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
                prev_info.loc[['MIN', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']] += row[['MIN', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']]
                seen_player[row['Player']] = prev_info, seen_player[row['Player']][1] + 1
            average_stats(df, index, seen_player, row)
        frames.append(df)

    playerDF = pd.concat(frames)
    pd.DataFrame(playerDF).to_csv("/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/NBA_Traditional_Season_Average_Stats_2.csv", header=['Player', 'Team', 'Match Up', 'Game Date', 'W/L', 'MIN', 'PTS', 'FG%', '3P%', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-'], index=None)
        
def collect_all_advanced_stats():
    playerDF= pd.DataFrame(columns=['Player', 'Team', 'Match Up', 'Game Date', 'W/L', 'MIN', 'OFF_RATING','DEF_RATING','NET_RATING','AST%','AST_TO','AST_RATIO','OREB%','DREB%','REB%','TM_TOV%','EFG%','TS%','USG%','PACE'])
    astypeDict= {'Player':str, 'Team':str, 'Match Up':str, 'Game Date':str, 'W/L':str, 'MIN':float, 'OFF_RATING':float, 'DEF_RATING':float, 'NET_RATING':float, 'AST%':float, 'AST_TO':float, 'AST_RATIO':float, 'OREB%':float, 'DREB%':float, 'REB%':float, 'TM_TOV%':float, 'EFG%':float, 'TS%':float, 'USG%':float, 'PACE':float}
    # groupingDict= {'Player':'first', 'Team': 'first', 'Match Up':'first', 'Game Date':'first', 'W/L':'first', 'MIN':'sum', 'PTS':'sum', 'FG%':'sum', '3P%':'sum', 'FT%':'sum', 'OREB':'sum', 'DREB':'sum', 'AST':'sum', 'STL':'sum', 'BLK':'sum', 'TOV':'sum', 'PF':'sum', '+/-':'sum'}
    frames=[]
    for year in tqdm(range(8, 22)):
        print(year)
        df = pd.read_csv('/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/AdvancedPlayerStats'+str(year//10) + str(year%10) + '-'+ str((year+1)//10) + str((year+1)%10) +'.csv')

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
                prev_info.loc[['MIN', 'OFF_RATING','DEF_RATING','NET_RATING','AST%','AST_TO','AST_RATIO','OREB%','DREB%','REB%','TM_TOV%','EFG%','TS%','USG%','PACE']] += row[['MIN', 'OFF_RATING','DEF_RATING','NET_RATING','AST%','AST_TO','AST_RATIO','OREB%','DREB%','REB%','TM_TOV%','EFG%','TS%','USG%','PACE']]
                seen_player[row['Player']] = prev_info, seen_player[row['Player']][1] + 1
            average_stats(df, index, seen_player, row)
        frames.append(df)

    playerDF = pd.concat(frames)
    pd.DataFrame(playerDF).to_csv("/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/NBA_Advanced_Season_Average_Stats_2.csv", header=['Player', 'Team', 'Match Up', 'Game Date', 'W/L', 'MIN', 'OFF_RATING','DEF_RATING','NET_RATING','AST%','AST_TO','AST_RATIO','OREB%','DREB%','REB%','TM_TOV%','EFG%','TS%','USG%','PACE'], index=None)
        
def create_dataset(flag):
    if flag == "traditional":
        df = pd.read_csv('/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/NBA_Traditional_Season_Average_Stats.csv')
        columns = ["Match Up", "Game Date", "W/L"]
        for j in range(2):
            for i in range(1, 9):
                pi_pts = "P" + str(i) + " PTS"
                pi_fg_pct = "P" + str(i) + " FG%"
                pi_3p_pct = "P" + str(i) + " 3P%"
                pi_ft_pct = "P" + str(i) + " FT%"
                pi_oreb = "P" + str(i) + " OREB"
                pi_dreb = "P" + str(i) + " DREB"
                pi_reb = "P" + str(i) + " REB"
                pi_ast = "P" + str(i) + " AST"
                pi_stl = "P" + str(i) + " STL"
                pi_blk = "P" + str(i) + " BLK"
                pi_tov = "P" + str(i) + " TOV"
                pi_pf = "P" + str(i) + " PF"
                pi_plus_minus = "P" + str(i) + " +/-"
                if j == 0:
                    team = "H "
                else:
                    team = "V "
                columns.extend([team + pi_pts, team + pi_fg_pct, team + pi_3p_pct, team + pi_ft_pct, team + pi_oreb, team + pi_dreb, team + pi_reb, team + pi_ast, team + pi_stl, team + pi_blk, team + pi_tov, team + pi_pf, team + pi_plus_minus])
        X = pd.DataFrame(columns=columns)
        game_dates = df["Game Date"].unique()
        game_dates = game_dates[len(game_dates)//2:]
        for date in tqdm(game_dates):
            cur_games = df[(df["Game Date"]==date)]
            match_ups = cur_games["Match Up"].unique()
            match_ups = [
                match_ups[index] for index in range(len(match_ups))
                if 'vs.' in match_ups[index]
            ]
            for match_up in match_ups:
                home_team_name = match_up[:3]
                visit_team_name = match_up[-3:]
                
                home_team = cur_games[(cur_games["Team"]==home_team_name)].sort_values(by=["MIN"], ascending=[False])[:8]
                visit_team = cur_games[(cur_games["Team"]==visit_team_name)].sort_values(by=["MIN"], ascending=[False])[:8]
                w_l = home_team["W/L"].unique()
                if w_l[0] == "L":
                    w_l = 0
                else:
                    w_l = 1
                row = {"Match Up": match_up, "Game Date": date, "W/L": w_l}
                player_number = 1
                for index, player in home_team.iterrows():
                    row["H P" + str(player_number) + " PTS"], row["H P" + str(player_number) + " FG%"], row["H P" + str(player_number) + " 3P%"], row["H P" + str(player_number) + " FT%"], row["H P" + str(player_number) + " OREB"], row["H P" + str(player_number) + " DREB"], row["H P" + str(player_number) + " REB"], row["H P" + str(player_number) + " AST"], row["H P" + str(player_number) + " STL"], row["H P" + str(player_number) + " BLK"], row["H P" + str(player_number) + " TOV"], row["H P" + str(player_number) + " PF"], row["H P" + str(player_number) + " +/-"] = [player["PTS"]], [player["FG%"]], [player["3P%"]], [player["FT%"]], [player["OREB"]], [player["DREB"]], [player["REB"]], [player["AST"]], [player["STL"]], [player["BLK"]], [player["TOV"]], [player["PF"]], [player["+/-"]]
                    player_number += 1
                player_number = 1
                for index, player in visit_team.iterrows():
                    row["V P" + str(player_number) + " PTS"], row["V P" + str(player_number) + " FG%"], row["V P" + str(player_number) + " 3P%"], row["V P" + str(player_number) + " FT%"], row["V P" + str(player_number) + " OREB"], row["V P" + str(player_number) + " DREB"], row["V P" + str(player_number) + " REB"], row["V P" + str(player_number) + " AST"], row["V P" + str(player_number) + " STL"], row["V P" + str(player_number) + " BLK"], row["V P" + str(player_number) + " TOV"], row["V P" + str(player_number) + " PF"], row["V P" + str(player_number) + " +/-"] = [player["PTS"]], [player["FG%"]], [player["3P%"]], [player["FT%"]], [player["OREB"]], [player["DREB"]], [player["REB"]], [player["AST"]], [player["STL"]], [player["BLK"]], [player["TOV"]], [player["PF"]], [player["+/-"]]
                    player_number += 1        
                row = pd.DataFrame(row)
                X = pd.concat(([X, row]), ignore_index=True)
        pd.DataFrame(X).to_csv("data/NBA_Traditional_Stats_Dataset_second_half.csv", index=None)    
    
    elif flag == "advanced":
        df = pd.read_csv('/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/NBA_Advanced_Season_Average_Stats.csv')
        columns = ["Match Up", "Game Date", "W/L"]
        for j in range(2):
            for i in range(1, 9):
                pi_off_rating = "P" + str(i) + " OFF_RATING"
                pi_def_rating = "P" + str(i) + " DEF_RATING"
                pi_net_rating = "P" + str(i) + " NET_RATING"
                pi_ast_pct = "P" + str(i) + " AST%"
                pi_ast_to = "P" + str(i) + " AST_TO"
                pi_ast_ratio = "P" + str(i) + " AST_RATIO"
                pi_oreb_pct = "P" + str(i) + " OREB%"
                pi_dreb_pct = "P" + str(i) + " DREB%"
                pi_reb_pct = "P" + str(i) + " REB%"
                pi_tm_tov_pct = "P" + str(i) + " TM_TOV%"
                pi_efg_pct = "P" + str(i) + " EFG%"
                pi_ts_pct = "P" + str(i) + " TS%"
                pi_usg_pct = "P" + str(i) + " USG%"
                pi_pace = "P" + str(i) + " PACE"
                if j == 0:
                    team = "H "
                else:
                    team = "V "
                columns.extend([team + pi_off_rating, team + pi_def_rating, team + pi_net_rating, team + pi_ast_pct, team + pi_ast_to, team + pi_ast_ratio, team + pi_oreb_pct, team + pi_dreb_pct, team + pi_reb_pct, team + pi_tm_tov_pct, team + pi_efg_pct, team + pi_ts_pct, team + pi_usg_pct, team + pi_pace])
        X = pd.DataFrame(columns=columns)
        game_dates = df["Game Date"].unique()
        game_dates = game_dates[:len(game_dates)//2]
        for date in tqdm(game_dates):
            cur_games = df[(df["Game Date"]==date)]
            match_ups = cur_games["Match Up"].unique()
            match_ups = [
                match_ups[index] for index in range(len(match_ups))
                if 'vs.' in match_ups[index]
            ]
            for match_up in match_ups:
                home_team_name = match_up[:3]
                visit_team_name = match_up[-3:]
                
                home_team = cur_games[(cur_games["Team"]==home_team_name)].sort_values(by=["MIN"], ascending=[False])[:8]
                visit_team = cur_games[(cur_games["Team"]==visit_team_name)].sort_values(by=["MIN"], ascending=[False])[:8]
                w_l = home_team["W/L"].unique()
                if w_l[0] == "L":
                    w_l = 0
                else:
                    w_l = 1
                row = {"Match Up": match_up, "Game Date": date, "W/L": w_l}
                player_number = 1
                for index, player in home_team.iterrows():
                    row["H P" + str(player_number) + " OFF_RATING"], row["H P" + str(player_number) + " DEF_RATING"], row["H P" + str(player_number) + " NET_RATING"], row["H P" + str(player_number) + " AST%"], row["H P" + str(player_number) + " AST_TO"], row["H P" + str(player_number) + " AST_RATIO"], row["H P" + str(player_number) + " OREB%"], row["H P" + str(player_number) + " DREB%"], row["H P" + str(player_number) + " REB%"], row["H P" + str(player_number) + " TM_TOV%"], row["H P" + str(player_number) + " EFG%"], row["H P" + str(player_number) + " TS%"], row["H P" + str(player_number) + " USG%"], row["H P" + str(player_number) + " PACE"] = [player["OFF_RATING"]], [player["DEF_RATING"]], [player["NET_RATING"]], [player["AST%"]], [player["AST_TO"]], [player["AST_RATIO"]], [player["OREB%"]], [player["DREB%"]], [player["REB%"]], [player["TM_TOV%"]], [player["EFG%"]], [player["TS%"]], [player["USG%"]], [player["PACE"]]
                    player_number += 1
                player_number = 1
                for index, player in visit_team.iterrows():
                    row["V P" + str(player_number) + " OFF_RATING"], row["V P" + str(player_number) + " DEF_RATING"], row["V P" + str(player_number) + " NET_RATING"], row["V P" + str(player_number) + " AST%"], row["V P" + str(player_number) + " AST_TO"], row["V P" + str(player_number) + " AST_RATIO"], row["V P" + str(player_number) + " OREB%"], row["V P" + str(player_number) + " DREB%"], row["V P" + str(player_number) + " REB%"], row["V P" + str(player_number) + " TM_TOV%"], row["V P" + str(player_number) + " EFG%"], row["V P" + str(player_number) + " TS%"], row["V P" + str(player_number) + " USG%"], row["V P" + str(player_number) + " PACE"] = [player["OFF_RATING"]], [player["DEF_RATING"]], [player["NET_RATING"]], [player["AST%"]], [player["AST_TO"]], [player["AST_RATIO"]], [player["OREB%"]], [player["DREB%"]], [player["REB%"]], [player["TM_TOV%"]], [player["EFG%"]], [player["TS%"]], [player["USG%"]], [player["PACE"]]
                    player_number += 1        
                row = pd.DataFrame(row)
                X = pd.concat(([X, row]), ignore_index=True)
        pd.DataFrame(X).to_csv("data/NBA_Advanced_Stats_Dataset_second_half.csv", index=None)
    print("Dataset created!")

if __name__ == '__main__':
    # collect_all_advanced_stats()

    # create_dataset("advanced")
    # x_1= pd.read_csv("data/NBA_Advanced_Stats_Dataset_first_half.csv")
    # x_2=pd.read_csv("data/NBA_Advanced_Stats_Dataset_second_half.csv")
    # X = pd.concat([x_1, x_2])
    # pd.DataFrame(X).to_csv("data/NBA_Advanced_Stats_Dataset.csv", index=None)


    #get labels
    X = pd.read_csv("csv file path here")
    Y = X.iloc[:, 2]