import requests
import numpy as np
import pandas as pd

TEAM_COLORS={
    "Arsenal": "#C3291E",
    "Luton": "#F78F1E",
    "West Ham": "#7A263A",
    "Chelsea":"#034694",
    "Nottingham Forest": "#DD0000",
    "Spurs": "#000000",
    "Manchester City":"#6CABDD",
    "Wolves": "#FDB913",
    "Liverpool": "#C8102E"
}

ATTACKER_METRICS = [
    'total_points',
    'goals_scored',
    'expected_goals', 
    'assists',
    'expected_assists', 
    'expected_goal_involvements', 
 #    'yellow_cards', 
#    'red_cards', 
#    'saves', 
    'bonus', 
    'bps', 
    'influence', 
    'creativity', 
    'threat', 
    'ict_index', 
    'starts', 
#    'value_season',
#    'value', 
#    'transfers_balance', 
    'selected'
]
DEFENDER_METRICS = [
    'total_points',
    'minutes', 
    'clean_sheets', 
    'goals_conceded', 
    'own_goals', 
    'yellow_cards', 
    'red_cards', 
    'bonus', 
    'bps', 
    'influence', 
    'creativity', 
    'threat', 
    'ict_index', 
    'starts', 
    'expected_goal_involvements', 
    'expected_goals_conceded', 
#    'value_season'
#    'value', 
#    'transfers_balance', 
#    'selected'
]

PLAYER_COLS = [
    'full_name',
    'team',
#    'element_type',
    'selected_by_percent',
    'now_cost',
 #   'minutes',
#    'transfers_in',
    'value_season',
    'total_points',
    'expected_goals',
    'goals_scored', 
    'expected_assists',
    'assists',
    'expected_goals_conceded',
    'goals_conceded'
]

PLAYER_IDS = {}

def get_season_df(season):
    def get_diff_difference(home_or_away, home_diff, away_diff):
        if home_or_away:
            return away_diff - home_diff
        else:
            return home_diff - away_diff
    gamedata = pd.read_csv(f'Fantasy-Premier-League/data/{season}/gws/' + 'merged_gw.csv')
    teams = pd.read_csv(f'Fantasy-Premier-League/data/{season}/' + 'teams.csv')
    fixtures = pd.read_csv(f'Fantasy-Premier-League/data/{season}/' + 'fixtures.csv')
    fixtures['team_h_name'] = fixtures['team_h'].apply(lambda x: teams.set_index('id').name[x])
    fixtures['team_a_name'] = fixtures['team_a'].apply(lambda x: teams.set_index('id').name[x])
    fixtures = fixtures[['id', 'team_a_name', 'team_h_name', 'team_a_difficulty', 'team_h_difficulty']]
    alldata = gamedata.merge(fixtures, left_on='fixture', right_on='id')
    alldata['difficulty_difference'] = alldata.apply(lambda x: get_diff_difference(x['was_home'], x['team_h_difficulty'], x['team_a_difficulty'] ), axis=1)
    return alldata

def get_tables():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'    
    r = requests.get(url)
    # Convert the response JSON content to a Python dictionary.
    json = r.json()
    
    # Create a DataFrame from the 'elements' section of the JSON, which contains player data.
    elements_df = pd.DataFrame(json['elements'])
    
    # Create a DataFrame from the 'element_types' section, containing player position types.
    elements_types_df = pd.DataFrame(json['element_types'])
    
    # Create a DataFrame from the 'teams' section, containing team data.
    teams_df = pd.DataFrame(json['teams'])
    
    # Map each player's position ID to its name from `elements_types_df`.
    elements_df['position'] = elements_df.element_type.map(elements_types_df.set_index('id').singular_name)
    
    # Map each player's team ID to its team name from `teams_df`.
    elements_df['team'] = elements_df.team.map(teams_df.set_index('id').name)
    
    # Convert the 'value_season' field to float for each player’s season value.
    elements_df['value'] = elements_df.value_season.astype(float)
    
    # Combine 'first_name' and 'second_name' columns to create the full player name.
    elements_df['full_name'] = elements_df['first_name'] + " " + elements_df['second_name']
    
    # Create a global dictionary `PLAYER_IDS` mapping each player's full name to their ID.
    globals()['PLAYER_IDS'] = elements_df[['full_name', 'id']].set_index('full_name').to_dict()['id']
    
    # Return the player and team DataFrames.
    return (elements_df, teams_df)


def player_photo(player_name, elements_df):
    image = (elements_df[elements_df['full_name']==player_name]['photo'].values[0].replace('jpg', 'png'))
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{image}"

def fetch_player_data(player_id):
    playerurl = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    r = requests.get(playerurl)
    json = r.json()
    return json

def player_past_metrics(player_id, metrics, start_gameweek=None, gameweeks=3, agg=None):
    json = fetch_player_data(player_id)
    gameweeks_available = len(json['history'])
    if start_gameweek==None:
        start_gameweek = gameweeks_available - gameweeks
    end_gameweek = start_gameweek + gameweeks
    result = {}
    for metric in metrics:
        metrics = [i[metric] for i in json['history'][start_gameweek:end_gameweek]]
        if agg == 'mean':
            if type(metrics[0]) == str:
                metrics = [float(i) for i in metrics]
            result[metric] = np.mean(metrics)
        elif agg == 'sum':
            if type(metrics[0]) == str:
                metrics = [float(i) for i in metrics]
            result[metric] = np.sum(metrics)
        elif agg is None:
            result[metric] = metrics
    return result

def get_best_players(elements_df, position=None):
    if position is not None:
        position_map = {"goalies":1, "defenders":2, "midfielders":3, "forwards":4}
        players = elements_df[(elements_df['minutes']>0) & (elements_df['element_type']==4)].sort_values(by='total_points', ascending=False)[PLAYER_COLS].head(20)
    forwards['difficulty'] = forwards['full_name'].apply(lambda x: np.mean(difficulty_range(PLAYER_IDS[x], gwrange=1)))

def compare_players(player_names, metrics, agg=None, start_gameweek=None, gameweeks=3):
    # Custom format function
    reverse_color_metrics = ["Difficulty", "Average Difficulty", "Goals Conceded", "Red Cards", "Yellow Cards", "Expected Goals Conceded"]  # Add metrics as needed

    def custom_format(val):
        if isinstance(val, (int, float)):
            return "{:.2f}".format(val)
        elif isinstance(val, list):
            return ', '.join(map(str, val))        
        return val

    def highlight_max(s):
        if type(s.iloc[0]) == list:
            s = s.apply(np.mean)
        is_max = s == s.max()
        color = 'rgba(0,200,0,0.2)' if s.name not in reverse_color_metrics else 'rgba(200,0,0,0.2)'
        # Check if all values are equal
        if s.nunique() == 1:
            return ['' for _ in s]
        return [f'background-color: {color}' if cell else '' for cell in is_max]
    def highlight_min(s):
        if type(s.iloc[0]) == list:
            s = s.apply(np.mean)
        is_min = s == s.min()
        color = 'rgba(200,0,0,0.2)' if s.name not in reverse_color_metrics else 'rgba(0,200,0,0.2)'
        # Check if all values are equal
        if s.nunique() == 1:
            return ['' for _ in s]
        return [f'background-color: {color}' if cell else '' for cell in is_min]

    columns = []
    for player_name in player_names:
        fixture_difficulty = difficulty_range(PLAYER_IDS[player_name], gwrange=1)
        player_metrics = player_past_metrics(PLAYER_IDS[player_name], metrics, agg=agg, start_gameweek=start_gameweek, gameweeks=gameweeks)
        if type(list(player_metrics.values())[0]) == list:
            player_metrics_values = [", ".join([str(j) for j in i]) for i in list(player_metrics.values())]
        else:
            player_metrics_values = list(player_metrics.values())
        player_metrics_values.insert(0, fixture_difficulty)
        
        columns.append(pd.Series(player_metrics_values, name=player_name))
    metrics_list = metrics.copy()
    metrics_list.insert(0, 'Difficulty')
    df = pd.DataFrame(
        data = {
            'Metric': metrics_list
        }
    )
    df = pd.concat(columns, axis=1)
    df.index = [i.replace("_", " ").title() for i in metrics_list]
    result = df.round(1)
    result.apply(highlight_max, axis=1)
    result = result.T
    result = result.style.apply(highlight_max, axis=0).apply(highlight_min, axis=0).format(custom_format)
    return result

def total_difficulty(player_id, start_gameweek=0, gameweeks=6):
    playerurl = 'https://fantasy.premierleague.com/api/element-summary/{}/'.format(player_id)
    r = requests.get(playerurl)
    json = r.json()
    end_gameweek = start_gameweek + gameweeks
    #print([i['difficulty'] for i in json['fixtures'][start_gameweek:end_gameweek]])
    return np.mean([i['difficulty'] for i in json['fixtures'][start_gameweek:end_gameweek]])

def difficulty_range(player_id, start_gameweek=0, end_gameweek=6, gwrange=3):
    result = []
    for i in range(end_gameweek):
        diff = total_difficulty(player_id, start_gameweek=i, gameweeks=gwrange)
        result.append(diff)
    return result

def format_chart(ax, xlabel=None, ylabel=None, title=None):
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=10)
    return ax

def compare_team(teams_df, metric, team, title=None, short_team_name=None):
    df = teams_df[[
        'name',
        metric
    ]].sort_values(by=metric, ascending=True)
    df = df.reset_index().drop('index', axis=1)
    if short_team_name is None:
        team_index = df[df['name']==team].index.values[0]
    else:
        team_index = df[df['name']==short_team_name].index.values[0]
    ax = df.set_index('name').plot(kind='barh', legend=False, figsize=(2,4), color='#ccc')
    ax = format_chart(ax, title=title)
    ax.tick_params(axis='y', length=0)
    ax.get_children()[team_index].set_color(TEAM_COLORS[team])
    return ax

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_weighted_score(df, weights, normalize=True, preferred_low=None):
    """
    Calculates a weighted score for each row in a DataFrame based on specified column weights,
    with consideration for columns where lower values are better.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns to be scored.
    - weights (dict): A dictionary where keys are column names and values are weights (1-2) to apply.
    - normalize (bool): If True, normalize the columns before applying weights (default is True).
    - preferred_low (list): A list of column names where lower values are better.
    
    Returns:
    - pd.DataFrame: The input DataFrame with an additional 'total_score' column, sorted by 'total_score' in descending order.
    """
    if preferred_low is None:
        preferred_low = []

    # Check if all specified weights are present in the DataFrame columns
    missing_cols = [col for col in weights.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")

    # Step 1: Optionally normalize the columns
    if normalize:
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df[weights.keys()]), columns=weights.keys())
    else:
        df_normalized = df[weights.keys()].copy()

    # Step 2: Adjust columns where lower values are preferred by inverting their scale
    for column in preferred_low:
        print(column)
        if column in df_normalized.columns:
            df_normalized[column] = 1 - df_normalized[column]  # Inverts normalized values so that lower is better

    # Step 3: Apply weights to each column
    for column, weight in weights.items():
        df_normalized[column] *= weight

    # Ensure the indices align with the original DataFrame
    df_normalized.index = df.index

    # Step 4: Calculate the total score by summing the weighted columns
    df['total_score'] = df_normalized.sum(axis=1)

    # Step 5: Sort by the total score and return the DataFrame with styling
    df_sorted = df.sort_values(by='total_score', ascending=False)
    df_sorted = style_dataframe(df_sorted)  # Apply styling for display
    
    return df_sorted



# Define global reverse color metrics for conditional formatting
reverse_color_metrics = ["Difficulty", "Average Difficulty", "Goals Conceded", "Red Cards", "Yellow Cards", "Expected Goals Conceded"]

def custom_format(val):
    """Format function to apply custom formatting to values."""
    if isinstance(val, (int, float)):
        return "{:.2f}".format(val)
    elif isinstance(val, list):
        return ', '.join(map(str, val))        
    return val

def highlight_max(s):
    """Highlight the maximum value in each column or row."""
    if isinstance(s.iloc[0], list):
        s = s.apply(np.mean)
    is_max = s == s.max()
    color = 'rgba(0,200,0,0.2)' if s.name not in reverse_color_metrics else 'rgba(200,0,0,0.2)'
    if s.nunique() == 1:
        return ['' for _ in s]
    return [f'background-color: {color}' if cell else '' for cell in is_max]

def highlight_min(s):
    """Highlight the minimum value in each column or row."""
    if isinstance(s.iloc[0], list):
        s = s.apply(np.mean)
    is_min = s == s.min()
    color = 'rgba(200,0,0,0.2)' if s.name not in reverse_color_metrics else 'rgba(0,200,0,0.2)'
    if s.nunique() == 1:
        return ['' for _ in s]
    return [f'background-color: {color}' if cell else '' for cell in is_min]

def style_dataframe(df):
    """
    Apply custom styling and formatting to a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to style.

    Returns:
    - pd.Styler: A styled DataFrame with custom formatting and highlighting.
    """
    return (
        df.style
        .apply(highlight_max, axis=0)
        .apply(highlight_min, axis=0)
        .format(custom_format)
    )
