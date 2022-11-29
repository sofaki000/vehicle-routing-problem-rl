# Define coordinates of where we want to center our map
import torch
from rl_for_solving_the_vrp.implementation_1 import config
from rl_for_solving_the_vrp.implementation_1.map import Map
from rl_for_solving_the_vrp.implementation_1.train import train_vrp
from rl_for_solving_the_vrp.implementation_1.vrp import VehicleRoutingDataset

map = Map(center=config.thessaloniki_coordinates, zoom_start=13)
map.create_map()
coords = map.plot_buffer_around_thessaloniki(config.num_nodes)
map.showMap(map_name="initial_map")

xs = []
ys = []
for i in range(len(coords[0])):
    x = coords[0][i]
    y = coords[1][i]
    xs.append(x)
    ys.append(y)


locations = torch.FloatTensor(coords)[None,:,:]

train_data = VehicleRoutingDataset(num_samples=config.train_size,
                                   input_size=config.num_nodes,
                                   max_load=20,
                                   max_demand=9,
                                   seed=None,
                                   locations = locations)

result_tour_indixes = train_vrp(train_data= train_data,
          valid_data=train_data,
          seed=config.seed,
          num_nodes=config.num_nodes,
          actor_lr=config.actor_lr,
          critic_lr=config.critic_lr,
          max_grad_norm=config.max_grad_norm,
          batch_size=config.batch_size,
          hidden_size=config.hidden_size,
          dropout=config.dropout,
          num_layers=config.layers,
          num_epochs=config.num_epochs)

locations = locations.squeeze(0).numpy()
xs = [locations[0][i] for i in result_tour_indixes][0][0]
ys = [locations[1][i] for i in result_tour_indixes][0][0]

print(f'Resulting tour is : {result_tour_indixes}')
number_of_stops = len(xs)
points_visiting = []

for i in range(number_of_stops):
    x = xs[i]
    y = ys[i]
    points_visiting.append([x, y ])

map.plot_lines_between_points(points_visiting, map_name="after_visiting")
# map.showMap( map_name="after_visiting",open=True)