import time
import webbrowser
import geopandas as gpd
import folium
import matplotlib.patches as patches
import numpy as np
from selenium.webdriver.chrome import webdriver
from shapely.geometry import Point
import os
from rl_for_solving_the_vrp.implementation_1 import config
import io
from PIL import Image

map_dir = 'C:\\Users\\Lenovo\\Desktop\\Διπλωματική\\vehicle-routing-problem-rl\\maps\\folium_map_utilities'
os.makedirs(map_dir, exist_ok=True)
map_name = f"{map_dir}/thessaloniki_map.html"

before_visiting_img_name = f"{map_dir}/initial_map.png"
after_visiting_map_name = f"{map_dir}/after_visiting_map"
after_visiting_map_name_html =  f"{after_visiting_map_name}.html"
after_visiting_map_name_png =  f"{map_dir}/{after_visiting_map_name}.png"

class Map:
    def __init__(self, center, zoom_start):
        self.center = center
        self.zoom_start = zoom_start
        self.map = folium.Map(location=config.thessaloniki_coordinates,
                              zoom_start=self.zoom_start,
                              tooltip='Map of thessaloniki'
                              )
        city_map = self.map
    def show_map(self, map_name, open_in_browser=False, save_png=False):
        # Display the map
        self.map.save(f'{map_name}.png')

        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(f'{map_name}.html'))
            #webbrowser.open(map_name)
        if save_png:
            img_data = self.map._to_png(10)
            img = Image.open(io.BytesIO(img_data))
            img.save(f'{map_name}.png')

    def add_markers_at_points(self,points):
        # expects an array of points: [ [x1,y1],[x2,y2],...]
        # first point is the depot
        xs = points[0]
        ys = points[1]
        for i in range(len(xs)):
            if i==0:
                # we denote depot
                folium.Marker([xs[i], ys[i]],icon=folium.Icon(color='green')).add_to(self.map)
            else:
                folium.Marker([xs[i], ys[i]]).add_to(self.map)

    def get_random_points_within_thessaloniki(self, number_of_nodes):
        thess_point = Point(config.thessaloniki_coordinates)
        thess_buffer = gpd.GeoSeries(thess_point).geometry.buffer(0.02)
        coords  = self.generate_random_location_within_ROI(number_of_nodes, thess_buffer)
        for i in range(len(coords[0])):
            x = coords[0][i]
            y = coords[1][i]
            folium.Marker([x,y]).add_to(self.map)

        img_data = self.map._to_png(10)
        img = Image.open(io.BytesIO(img_data))
        img.save(before_visiting_img_name)
        return coords
    def plot_lines_between_points(self, points, show_map=False):
        # for i in range(len(points)):
        #     lat = points[i][0]
        #     lon = points[i][1]
        #     folium.Marker([lat, lon], popup=(f'stop:{i}'),
        #                   icon=folium.Icon(color='green', icon='plus')).add_to(self.map)

        attr = {'fill': '#007DEF', 'font-weight': 'bold', 'font-size': '100'}
        colors = ["red", "green", "blue","#3388ff","#3389ff","#3322ff","red","green"]
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            print(f'road from p1:{p1} to p2 {p2}')
            folium.PolyLine([p1 ,p2],  "20 km",   color=colors[i],    center=True,  offset=7,  attributes=attr, no_clip =True).add_to(self.map)

        if show_map:
            self.map.save(after_visiting_map_name_html)

            webbrowser.open('file://' + os.path.realpath(after_visiting_map_name_html))
            # img_data = self.map._to_png(60)
            # img = Image.open(io.BytesIO(img_data))
            # img.save(f'{map_name}.png')
            # use selenium to save the html as png image
            self.map.save(after_visiting_map_name_html)
            mapUrl = 'file://{0}/{1}'.format(os.getcwd(), after_visiting_map_name_html)

            # from selenium import webdriver
            # driver = webdriver.Firefox()
            # driver.get(mapUrl)
            # # wait for 5 seconds for the folium_map_utilities and other assets to be loaded in the browser
            # time.sleep(5)
            # driver.save_screenshot(after_visiting_map_name_png)
            # driver.quit()

    def generate_random_location_within_ROI(self, num_pt, polygon):
      # define boundaries
      boundaries = polygon.bounds
      minx = min(boundaries.minx)
      maxx = max(boundaries.maxx)
      miny = min(boundaries.miny)
      maxy = max(boundaries.maxy)

      i = 0
      x = []
      y = []
      points = []
      while i < num_pt:
        # generate random location coordinates
        x_t = np.random.uniform(minx, maxx)
        y_t = np.random.uniform(miny, maxy)
        # further check whether it is in the city area
        for p in polygon:
          if Point(x_t, y_t).within(p):
            x.append(x_t)
            y.append(y_t)
            points.append(Point(x_t,y_t))
            i = i + 1
            break

      return x, y

#locations, loads, demands = get_excel_data(file_path=config.data_path)
#plot_locations_in_thessaloniki(locations)

def plot_locations_in_thessaloniki(locations):
    map = Map(center=config.thessaloniki_coordinates, zoom_start=12)
    map.add_markers_at_points(locations)
    map.show_map("initial_map_v2", open_in_browser=True, save_png=True)


def plot_locations_visited_map(locations, result_tour_indixes):
    locations = locations.squeeze(0).numpy()
    xs = [locations[0][i] for i in result_tour_indixes][0][0]
    ys = [locations[1][i] for i in result_tour_indixes][0][0]

    print(f'Resulting tour is : {result_tour_indixes}')
    number_of_stops = len(xs)
    points_visiting = []

    for i in range(number_of_stops):
        x = xs[i]
        y = ys[i]
        points_visiting.append([x, y])

    map = Map(center=config.thessaloniki_coordinates, zoom_start=13)
    map.plot_lines_between_points(points_visiting, show_map=True)
