import folium
import os
import json
import pandas as pd
from flask import Flask, render_template, redirect, url_for
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from branca.colormap import LinearColormap
from decimal import Decimal, getcontext
import ast

# Assuming you have your city_coordinates, transition_dict, and stationary_probabilities


# Read cities from cityList.txt and create a list
with open('cityList.txt', 'r') as file:
    cityList = [city.strip() for city in file if city.strip()]

# Read JSON data from my_dict.json and store it in cityConnections
with open('my_dict.json', 'r') as file:
    cityConnections = json.load(file)

def parse_coordinates(coord_string):
    try:
        coords = ast.literal_eval(coord_string)
        if isinstance(coords, tuple) and len(coords) == 1 and isinstance(coords[0], list):
            return coords[0]
        return coords
    except:
        return None

city_coordinates = {}

with open('cityCoordinates.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                city = parts[0].strip().strip("'")
                coords = parse_coordinates(parts[1].strip())
                if coords:
                    city_coordinates[city] = coords

# End of Get Data

# =============================================================================
def create_city_map(city, city_coordinates, transition_dict, stationary_probabilities):
    # Center the map on the focus city
    center_lat, center_lon = city_coordinates[city]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Create a color map based on transition probabilities
    transition_probs = list(transition_dict[city].values())
    min_prob = min(transition_probs)
    max_prob = max(transition_probs)
    colormap = LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=min_prob, vmax=max_prob)

    # Add markers for each city
    for other_city, coords in city_coordinates.items():
        stationary_prob = stationary_probabilities[other_city]
        transition_prob = transition_dict[city][other_city]
        color = colormap(transition_prob)
        size = 5 + (transition_prob - min_prob) / (max_prob - min_prob) * 15  # Scale size between 5 and 20

        folium.CircleMarker(
            location=coords,
            radius=size,
            popup=f"{other_city}<br>Transition Prob: {transition_prob:.4f}<br>Stationary Prob: {stationary_prob:.4f}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)

    # Highlight the focus city
    folium.Marker(
        location=[center_lat, center_lon],
        popup=city,
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(m)

    # Add a color bar legend
    colormap.add_to(m)

    return m

# Getting the Database from the CSV file

try:
    full_database = pd.read_csv('fullDatabase.csv')
    print(f"Successfully loaded fullDatabase.csv. Shape: {full_database.shape}")
except FileNotFoundError:
    print("Error: fullDatabase.csv not found.")
except pd.errors.EmptyDataError:
    print("Error: fullDatabase.csv is empty.")
except pd.errors.ParserError:
    print("Error: Unable to parse fullDatabase.csv. Please check the file format.")
except Exception as e:
    print(f"An unexpected error occurred while reading fullDatabase.csv: {str(e)}")




# =============================================================================


# Set the precision for Decimal output
getcontext().prec = 6

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def create_distance_matrix(city_coordinates):
    cities = list(city_coordinates.keys())
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lon1 = city_coordinates[cities[i]]
                lat2, lon2 = city_coordinates[cities[j]]
                distance_matrix[i, j] = haversine_distance(lat1, lon1, lat2, lon2)
            else:
                distance_matrix[i, j] = 10  # 100 meters for self-connections
    
    return distance_matrix, cities

def create_transition_matrix(distance_matrix, cities):
    probability_matrix = 1 / distance_matrix
    row_sums = probability_matrix.sum(axis=1)
    transition_matrix = probability_matrix / row_sums[:, np.newaxis]
    
    # Create nested dictionary structure
    transition_dict = {}
    for i, city in enumerate(cities):
        transition_dict[city] = {other_city: prob
                                 for other_city, prob in zip(cities, transition_matrix[i])}
    
    return transition_matrix, transition_dict

# Assuming city_coordinates is your dictionary of city coordinates
distance_matrix, cities = create_distance_matrix(city_coordinates)
transition_matrix, transition_dict = create_transition_matrix(distance_matrix, cities)

def calculate_stationary_distribution(transition_matrix, num_iterations=10000):
    n = transition_matrix.shape[0]
    distribution = np.ones(n) / n  # Start with a uniform distribution
    
    for _ in range(num_iterations):
        new_distribution = np.dot(distribution, transition_matrix)
        if np.allclose(distribution, new_distribution):
            break
        distribution = new_distribution
    
    return distribution

# Calculate the stationary distribution
stationary_distribution = calculate_stationary_distribution(transition_matrix)

# Create a dictionary to store the stationary probabilities for each city
stationary_probabilities = {city: prob for city, prob in zip(cities, stationary_distribution)}


def generate_all_maps(city_coordinates, transition_dict, stationary_probabilities):
    # Create a folder to store the maps
    os.makedirs('static/city_maps', exist_ok=True)

    # Generate maps for each city
    for city in city_coordinates.keys():
        print(f"Generating map for {city}")
        m = create_city_map(city, city_coordinates, transition_dict, stationary_probabilities)
        m.save(f'static/city_maps/{city.replace(" ", "_")}_map.html')

    print("All maps generated successfully.")


generate_all_maps(city_coordinates, transition_dict, stationary_probabilities)



# Set up Flask app
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    city_list = sorted(city_coordinates.keys())
    return render_template('index.html', cities=city_list)

@app.route('/city/<city>')
def city_map(city):
    filename = f'{city.replace(" ", "_")}_map.html'
    return redirect(url_for('static', filename=f'city_maps/{filename}'))

if __name__ == '__main__':
    app.run()