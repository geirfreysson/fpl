from fpl_helpers import *
import pandas as pd

def test_min_max():
  elements_df, teams_df = get_tables()
  global PLAYER_IDS
  PLAYER_IDS = elements_df[['full_name', 'id']].set_index('full_name').to_dict()['id']

  df = compare_players(['Bukayo Saka', 'Sven Botman'], metrics=ATTACKER_METRICS, agg='mean')

  print(df)