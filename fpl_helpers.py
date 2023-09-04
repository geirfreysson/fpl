import requests
import numpy as np
import pandas as pd

TEAM_COLORS={
    "Arsenal": "#C3291E",
    "Luton": "#F78F1E",
    "West Ham": "#7A263A",
    "Chelsea":"#034694",
    "Nottingham Forest": "#DD0000",
    "Spurs": "#000000"
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
#    'value', 
#    'transfers_balance', 
#    'selected'
]
DEFENDER_METRICS = [
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

def get_tables():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()
    elements_df = pd.DataFrame(json['elements'])
    elements_types_df = pd.DataFrame(json['element_types'])
    teams_df = pd.DataFrame(json['teams'])
    elements_df['position'] = elements_df.element_type.map(elements_types_df.set_index('id').singular_name)
    elements_df['team'] = elements_df.team.map(teams_df.set_index('id').name)
    elements_df['value'] = elements_df.value_season.astype(float)
    elements_df['full_name'] = elements_df['first_name'] + " " +  elements_df['second_name']
    #global PLAYER_IDS
    globals()['PLAYER_IDS'] = elements_df[['full_name', 'id']].set_index('full_name').to_dict()['id']
    return (elements_df, teams_df)

def player_photo(player_name, elements_df):
    image = (elements_df[elements_df['full_name']==player_name]['photo'].values[0].replace('jpg', 'png'))
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{image}"

def fetch_player_data(player_id):
    playerurl = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    r = requests.get(playerurl)
    json = r.json()
    return json

def player_past_metrics(player_id, metrics, start_gameweek=0, gameweeks=6, agg=None):
    json = fetch_player_data(player_id)
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

def compare_players(player_names, metrics, agg=None):
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: rgba(0,200,0,0.2)' if cell else '' for cell in is_max]

    columns = []
    for player_name in player_names:
        player_metrics = player_past_metrics(PLAYER_IDS[player_name], metrics, agg=agg)
        if type(list(player_metrics.values())[0]) == list:
            player_metrics_values = [", ".join([str(j) for j in i]) for i in list(player_metrics.values())]
        else:
            player_metrics_values = list(player_metrics.values())
        columns.append(pd.Series(player_metrics_values, name=player_name))
    df = pd.DataFrame(
        data = {
            'Metric': metrics
        }
    )
    df = pd.concat(columns, axis=1)
    df.index = [i.replace("_", " ").title() for i in metrics]
    return df.round(1).style.apply(highlight_max, axis=1).format("{:.2f}")

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