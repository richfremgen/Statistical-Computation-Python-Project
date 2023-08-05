import zipfile
import pandas as pd
from io import BytesIO

id_comp = pd.DataFrame()

with zipfile.ZipFile("data/events.zip") as z:
    for filename in z.namelist():
        print(f'Importing {filename}')
        with z.open(filename) as file:
            event = pd.read_json(BytesIO(file.read()))
            event = event[['id', 'teamId', 'eventName']]
            event = event[event['eventName'] == 'Shot']
            event = event.drop(['eventName'], axis=1)
            competition = filename.replace('.json', '').replace('events_', '')
            event['competition'] = competition
            id_comp = pd.concat([id_comp, event])
            
id_comp = id_comp.drop_duplicates().reset_index().drop('index', axis=1)
id_comp.to_pickle('data/processed/dt_teams.pkl')
