import pandas as pd
import numpy as np


def make_angle(x, y):
    x_area = 100*105/100
    y_area_low = 37*68/100
    y_area_up = 63 * 68 / 100

    b = np.abs((y_area_low - y_area_up))
    a = np.sqrt((x - x_area)**2 + (y - y_area_low)**2)
    c = np.sqrt((x - x_area) ** 2 + (y - y_area_up) ** 2)
    cos = -1/(2*a*c)*(b**2 - a**2 - c**2)

    return np.arccos(cos)


def make_distance(x, y):
    x_goal = 100 * 105 / 100
    y_goal = 50 * 68 / 100
    return np.sqrt((x - x_goal) ** 2 + (y - y_goal) ** 2)


def make_shot_df(event, tag_names):
    dt_shots = event[event['eventName'] == 'Shot']
    dt_player = dt_shots[['id', 'playerId']]
    dt_shots = dt_shots.explode('tags', ignore_index=True)
    # Convert tags to a value
    dt_shots['tags'] = dt_shots['tags'].apply(pd.Series)
    dt_shots = dt_shots.explode('positions')
    # dt_shots['dummy'] = np.tile(['start', 'end'], 65390 // 2)
    dt_shots['dummy'] = np.tile(['start', 'end'], dt_shots.shape[0] // 2)
    dt_shots = pd.concat([
        dt_shots.drop(['positions', 'dummy'], axis=1),
        dt_shots.pivot(columns='dummy', values='positions')
    ], axis=1)
    dt_shots['start_x'] = pd.Series([x['x'] for x in dt_shots.start])
    dt_shots['start_y'] = pd.Series([x['y'] for x in dt_shots.start])
    dt_shots['end_x'] = pd.Series([x['x'] for x in dt_shots.end])
    dt_shots['end_y'] = pd.Series([x['y'] for x in dt_shots.end])
    dt_shots = dt_shots.drop(['start', 'end'], axis=1).drop_duplicates(ignore_index=True)
    # Convert (x,y) from percentage of pitch to coordinates (pitch is 105x68 meters)
    # dt_shots = dt_shots.loc[(dt_shots['start_x'] != 90) | (dt_shots['start_y'] != 50)]
    dt_shots[['start_x', 'end_x']] = dt_shots[['start_x', 'end_x']].apply(lambda x: x * 105.0 / 100.0)
    dt_shots[['start_y', 'end_y']] = dt_shots[['start_y', 'end_y']].apply(lambda x: x * 68.0 / 100.0)
    dt_shots = dt_shots.merge(tag_names, on='tags', how='left').drop(['tags'], axis=1)

    dt_shots = dt_shots.drop(
        [
            'eventId', 'subEventName', 'playerId',
            'matchId', 'eventName', 'teamId', 'matchPeriod',
            'eventSec', 'subEventId'
        ],
        axis=1
    )

    dt_shots = pd.get_dummies(dt_shots, columns=['label'])
    dt_positions = dt_shots[['id', 'start_x', 'start_y']]. \
        groupby('id'). \
        agg('mean'). \
        reset_index()

    dt_labels = dt_shots. \
        drop(['start_x', 'start_y', 'end_x', 'end_y'], axis=1). \
        groupby('id'). \
        agg('sum'). \
        reset_index()

    dt_shots_clean = dt_positions.merge(dt_labels, on='id', how='left')
    dt_shots_clean = dt_shots_clean.merge(dt_player, on='id', how='left')
    dt_shots_clean['angle'] = make_angle(dt_shots_clean.start_x, dt_shots_clean.start_y)
    dt_shots_clean['distance'] = make_distance(dt_shots_clean.start_x, dt_shots_clean.start_y)

    return dt_shots_clean
