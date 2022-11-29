import pandas as pd
import torch

from rl_for_solving_the_vrp.implementation_1 import config
from rl_for_solving_the_vrp.implementation_1.map import Map

df = pd.read_excel(config.data_path)

df = pd.DataFrame(df)
latitude_x = df.iloc[:,1]
longtitude_y = df.iloc[:,2]
demands = df.iloc[:,3]
load = df.iloc[:,4][0]
map = Map(center=config.thessaloniki_coordinates, zoom_start=13)
map.create_map()

locations = torch.FloatTensor([latitude_x, longtitude_y])[None,:,:]
map.add_markers_at_points([latitude_x, longtitude_y])
map.showMap("initial_map.html", open_in_browser=True, save_png=False)