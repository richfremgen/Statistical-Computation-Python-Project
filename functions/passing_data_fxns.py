# import packages
import pandas as pd
import json
import glob
import zipfile
from io import BytesIO
import numpy as np

# read data files from /data
def read_data():
    # teams 
    print('Starting')
    with open("data/teams.json", encoding="unicode_escape") as f:  
        data = f.read()  
        teams = pd.read_json(data)
        
    # competitions 
    with open("data/competitions.json", encoding="unicode_escape") as f:  
        data = f.read()  
        comp = pd.read_json(data)
        
    # players 
    with open("data/players.json", encoding="unicode_escape") as f:  
        data = f.read()  
        players = pd.read_json(data)
        
    # matches 
    # for everything in matches folder, read in and append together 
    matches_list = glob.glob("data/matches/*.json")

    matches = pd.DataFrame()
    for name in matches_list:
        with open(name, encoding="unicode_escape") as f:  
            data = f.read()  
            match = pd.read_json(data)
            matches = pd.concat([matches,match])

    # events, zipped files
    # for everything in events zip, read in and append together 
    events = pd.DataFrame()
    with zipfile.ZipFile('data/events.zip', "r") as z:
        for filename in z.namelist(): 
            with z.open(filename) as f:  
                data = f.read()  
                event = pd.read_json(BytesIO(data))
                events = pd.concat([events,event])

    # tags data
    tag_df = pd.read_csv('data/tags2name.csv').drop(
        ['Description'],
        axis=1
    ).rename(
        columns={'Tag': 'tags', 'Label': 'label'}
    )
    
    print('Finished reading data')
    return teams, comp, players, matches, events, tag_df


# clean data 
def clean_data(teams, comp, players, matches, events, tag_df):
    # teams 
    # filter for only club and needed columns 
    teams_c = teams[teams.type == 'club'][['officialName', 'wyId', 'area']].copy()
    # get just club name from area column, rename cols
    teams_c = pd.concat([teams_c.drop(['area'], axis=1), teams_c.area.apply(pd.Series).name], axis = 1
                       ).rename({'officialName': 'club_name', 
                                 'wyId':'team_id', 
                                 'name': 'country'}, axis=1)

    # comp 
    comp_c = comp[['name', 'wyId']].copy().rename({'name':'league', 'wyId':'league_id'}, axis=1)

    # players 
    players_c = players[['wyId', 'role']].copy()
    # get only position name from role column
    players_c = pd.concat([players_c.drop(['role'], axis=1), players_c.role.apply(pd.Series).name], axis = 1
                         ).rename({'wyId': 'player_id', 'name': 'position'}, axis=1)

    # matches 
    matches_c = matches[matches.status == 'Played'].copy(
                ).rename(
                    {'wyId': 'match_id',
                     'roundId':'round_id',
                     'teamsData':'teams_data',
                     'seasonId':'season_id',
                     'winner':'winner_team_id',
                     'competitionId':'comp_id'}, axis=1)

    # events
    events_c = events.copy().rename({'eventId': 'event_type', 
                   'playerId':'player_id', 
                   'matchId':'match_id',
                  'subEventName':'sub_event_type',
                  'eventName':'event_name',
                  'teamId':'team_id',
                  'matchPeriod':'match_period', 
                  'eventSec':'event_sec',
                  'subEventId':'sub_event_id',
                  'id':'event_id'}, axis=1)
    
    print('Finished cleaning data')
    return teams_c, comp_c, players_c, matches_c, events_c


# functions for passing section
def get_pass_shot_data():
    teams, comp, players, matches, events, tag_df = read_data()
    teams_c, comp_c, players_c, matches_c, events_c = clean_data(teams, comp, players, matches, events, tag_df)
    
    # filter for just shot/pass data 
    ps_df = events_c[events_c.event_name.isin(['Shot', 'Pass'])].copy()

    # making position columns separately
    pos = pd.DataFrame(ps_df.positions.tolist())
    pos.rename({0:'start', 1:'end'}, axis =1, inplace=True)
    pos = pos[pos.end.notnull()]
    index = pos.index
    origin_pos = pd.DataFrame(pos.start.tolist()).rename({'y':'origin_y', 'x':'origin_x'}, axis=1)          
    dest_pos = pd.DataFrame(pos.end.tolist()).rename({'y':'dest_y', 'x':'dest_x'}, axis=1)
    pos_df = pd.concat([origin_pos, dest_pos], axis=1)
    
    # concat back to ps_df
    ps_df = pd.concat([ps_df.iloc[index].reset_index(), pos_df], axis =1)
    # make coordinates 
    ps_df[['origin_x', 'dest_x']] = ps_df[['origin_x', 'dest_x']].apply(lambda x: x * 105 / 100)
    ps_df[['origin_y', 'dest_y']] = ps_df[['origin_y', 'dest_y']].apply(lambda x: x * 68 / 100)
    # fix position for shots
    ps_df.loc[ps_df.dest_x == 0, 'dest_x'] = 100 * 105 / 100
    ps_df.loc[ps_df.dest_y == 0, 'dest_y'] = 50 * 68 / 100
    # make distance
    ps_df['distance'] = np.sqrt((ps_df.origin_x - ps_df.dest_x)**2 + 
                                       (ps_df.origin_y - ps_df.dest_y)**2)

    print('Finished making position columns')
    
    # tags 
    # ps_df = ps_df.explode('tags', ignore_index=True)
    # ps_df['tags'] = pd.DataFrame(ps_df.tags.tolist())

    # merge id columns together 
    ps_df = ps_df.merge(matches_c[['winner_team_id','match_id', 'comp_id']], on='match_id'
                  ).merge(teams_c, on='team_id').merge(players_c, on='player_id'
                    ).merge(teams_c[['team_id','club_name']], left_on='winner_team_id', right_on='team_id'
                           ).rename({'club_name_x':'club_name', 'club_name_y':'winner'}, axis = 1)
    # .merge(tag_df, on='tags')
    
    print('Finished merging data')

    # make wales england, monaco france 
    ps_df.loc[ps_df.country == 'Wales', 'country'] = 'England'
    ps_df.loc[ps_df.country == 'Monaco', 'country'] = 'France'

    # remove unneeded columns 
    ps_df.drop(['tags', 'team_id_x', 'team_id_y', 'winner_team_id', 'positions', 'index'], axis=1, inplace=True)
    
    return ps_df


def get_summary_pass_data(df, num_cuts):
    cuts = pd.DataFrame({str(feature) + 'Bin' : pd.cut(df[feature], num_cuts) for feature in ['origin_x', 'origin_y']})
    x_bins = cuts.origin_xBin.unique()
    y_bins = cuts.origin_yBin.unique()
    
    # subset necesary data 
    pp_avg = df[['event_id','event_name','origin_x','origin_y', 'dest_x','dest_y', 'country', 'position']].copy()

    # make empty col first 
    pp_avg['or_avg_x'] = None
    pp_avg['or_avg_y'] = None
    pp_avg['dest_avg_x'] = None
    pp_avg['dest_avg_y'] = None


    # lump pass position together
    for i in range(len(x_bins)):
        pp_avg.loc[(pp_avg.origin_x >= x_bins[i].left) & 
                 (pp_avg.origin_x <= x_bins[i].right), 'or_avg_x'] = x_bins[i].mid
        pp_avg.loc[(pp_avg.origin_y >= y_bins[i].left) & 
                 (pp_avg.origin_y <= y_bins[i].right), 'or_avg_y'] = y_bins[i].mid

        pp_avg.loc[(pp_avg.dest_x >= x_bins[i].left) & 
                 (pp_avg.dest_x <= x_bins[i].right), 'dest_avg_x'] = x_bins[i].mid
        pp_avg.loc[(pp_avg.dest_y >= y_bins[i].left) & 
                 (pp_avg.dest_y <= y_bins[i].right), 'dest_avg_y'] = y_bins[i].mid


    # make position in terms of draw pitch coords
    pp_avg['or_avg_x'] = pp_avg['or_avg_x'] / 105 * 120 
    pp_avg['dest_avg_x'] = pp_avg['dest_avg_x'] / 105 * 120 
    pp_avg['or_avg_y'] = pp_avg['or_avg_y'] / 68 * 90
    pp_avg['dest_avg_y'] = pp_avg['dest_avg_y'] / 68 * 90

    return pp_avg

    
    