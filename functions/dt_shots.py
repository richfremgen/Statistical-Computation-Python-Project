import pandas as pd
import zipfile
from io import BytesIO
from main import make_shot_df

tag_names = pd.read_csv('data/tags2name.csv').drop(
    ['Description'],
    axis=1
).rename(
    columns={'Tag': 'tags', 'Label': 'label'}
)

df_shots = pd.DataFrame()

with zipfile.ZipFile("data/events.zip") as z:
    for filename in z.namelist():
        print(f'Importing {filename}')
        with z.open(filename) as file:
            event = pd.read_json(BytesIO(file.read()))
            shots_comp = make_shot_df(event, tag_names)
            df_shots = pd.concat([df_shots, shots_comp])

df_shots.head()
df_shots.to_pickle('data/processed/dt_shots.pkl')
