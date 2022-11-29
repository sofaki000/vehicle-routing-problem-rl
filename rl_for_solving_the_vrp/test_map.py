import webbrowser

import folium as folium
import geopandas as gpd
from folium import Marker, GeoJson
from folium.plugins import HeatMap

from rl_for_solving_the_vrp.implementation_1 import config
from rl_for_solving_the_vrp.implementation_1.map_utilities import Map

releases = gpd.read_file("data/toxic_release_pennsylvania.shp")
releases.head()

print(releases.crs)

# Select one release incident in particular
recent_release = releases.iloc[360]

# Measure distance from release to each station
distances = releases.geometry.distance(recent_release.geometry)
print(distances)


print('Mean distance to monitoring stations: {} feet'.format(distances.mean()))

print('Closest monitoring station ({} feet):'.format(distances.min()))
print((releases.iloc[distances.idxmin()]))
world_cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
two_mile_buffer = releases.geometry.buffer(2*5280)
print(two_mile_buffer.head())

# Create map with release incidents and monitoring stations
m = folium.Map(location=[39.9526, -75.1652], zoom_start=11)
# HeatMap(data=releases[['LATITUDE', 'LONGITUDE']], radius=15).add_to(m)
# for idx, row in releases.iterrows():
#     Marker([row['LATITUDE'], row['LONGITUDE']]).add_to(m)
thess_latitude = "40° 38' 24.2268'' N"
thess_longitude ="22° 56' 39.9084'' E"

# Plot each polygon on the map
GeoJson(two_mile_buffer.to_crs(epsg=4326)).add_to(m)
folium.Marker( [24.2268,39.9084]).add_to(m)


def get_buffer_around_point(point):
    two_mile_buffer = point.geometry.buffer(2 * 5280)
    two_mile_buffer.plot()


def plot_greece():
    #athens = world_cities[world_cities.name == "Athens"]
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    world_cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
    world.plot(column='gdp_per_cap');
    greece = world.loc[world['name'] == 'Greece']
    boundaries = greece['geometry']
    boundaries.plot()


def plot_two_miles_around_boundaries(boundaries):
    two_mile_buffer = boundaries.geometry.buffer(2 * 5280)
    two_mile_buffer.plot()

