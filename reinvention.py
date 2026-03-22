import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import os
import time
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np
import math

# ---------------- LOAD MODEL ----------------
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(base_path, "columns.pkl"), "rb"))

# ---------------- SESSION ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---------------- PAGE ----------------
st.set_page_config(page_title="Smart Delivery System", layout="wide")
st.title("🚚 Smart Delivery Routing System")

left, right = st.columns([3, 1])

# ---------------- INPUT ----------------
with right:
    st.subheader("📦 Setup")

    pickup_lat = st.number_input("Pickup Latitude", value=28.6)
    pickup_lon = st.number_input("Pickup Longitude", value=77.2)

    num_stops = st.number_input("Stops", 1, 5, 2)

    num_vehicles = st.number_input("Vehicles", 1, 5, 1)
    vehicle_capacity = st.number_input("Capacity per vehicle", 1, 50, 10)

    delivery_points = []
    demands = []

    for i in range(num_stops):
        lat = st.number_input(f"Lat {i+1}", key=f"lat{i}")
        lon = st.number_input(f"Lon {i+1}", key=f"lon{i}")
        demand = st.number_input(f"Demand {i+1}", 1, 10, 1, key=f"d{i}")

        delivery_points.append((lat, lon))
        demands.append(demand)

    weather = st.selectbox("Weather", ["clear","rainy","foggy","hot","cold","stormy"])
    vehicle = st.selectbox("Vehicle", ["Bike","Car","Truck"])

# ---------------- HAVERSINE ----------------
def haversine(p1, p2):
    R = 6371
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# ---------------- MATRIX ----------------
def compute_matrix(points):
    size = len(points)
    matrix = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = int(haversine(points[i], points[j]) * 1000)
    return matrix

# ---------------- OPTIMIZATION ----------------
def optimize(points, demands, vehicle_capacity, num_vehicles, vehicle):

    manager = pywrapcp.RoutingIndexManager(len(points), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    matrix = compute_matrix(points)

    vehicle_speed = {"Bike":30, "Car":50, "Truck":40}

    def distance_cb(i, j):
        dist = matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
        speed_factor = 50 / vehicle_speed[vehicle]
        return int(dist * speed_factor)

    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # CAPACITY CONSTRAINT
    def demand_cb(i):
        node = manager.IndexToNode(i)
        return demands[node-1] if node != 0 else 0

    demand_callback = routing.RegisterUnaryTransitCallback(demand_cb)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0,
        [vehicle_capacity]*num_vehicles,
        True,
        "Capacity"
    )

    # DISTANCE LIMIT
    routing.AddDimension(
        transit,
        0,
        20000,
        True,
        "Distance"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)

    routes = []
    if solution:
        for v in range(num_vehicles):
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            routes.append(route)

    return routes

# ---------------- ROUTE FETCH ----------------
@st.cache_data
def fetch_route(start, end):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
        res = requests.get(url, timeout=5)
        return res.json()["routes"][0]
    except:
        return None

# ---------------- MAP ----------------
with left:
    st.subheader("🗺️ Map")

    delivery_points = [p for p in delivery_points if p != (0.0, 0.0)]
    points = [(pickup_lat, pickup_lon)] + delivery_points

    total_distance = 0
    route_coords = []

    if len(points) > 1:
        routes = optimize(points, demands, vehicle_capacity, num_vehicles, vehicle)
    else:
        routes = []

    folium_map = folium.Map(location=[pickup_lat, pickup_lon], zoom_start=11)

    colors = ["blue","green","red","purple","orange"]

    for v_idx, route in enumerate(routes):
        if not route:
            continue

        ordered_points = [points[i] for i in route]
        color = colors[v_idx % len(colors)]

        for i in range(len(ordered_points)-1):
            r = fetch_route(ordered_points[i], ordered_points[i+1])

            if r:
                coords = r["geometry"]["coordinates"]
                dist = r["distance"]/1000
                total_distance += dist

                latlon = [[c[1],c[0]] for c in coords]
                route_coords.extend(latlon)

                folium.PolyLine(latlon, color=color, weight=5).add_to(folium_map)
            else:
                folium.PolyLine([ordered_points[i], ordered_points[i+1]], color="red").add_to(folium_map)

        for idx, p in enumerate(ordered_points):
            folium.Marker(p, tooltip=f"V{v_idx+1}-Stop{idx}").add_to(folium_map)

    if delivery_points:
        HeatMap(delivery_points).add_to(folium_map)

    st_folium(folium_map, width=800)

# ---------------- ETA ----------------
traffic_factor = {"clear":1,"rainy":1.3,"foggy":1.2,"stormy":1.5,"hot":1.1,"cold":1.1}
vehicle_speed = {"Bike":30,"Car":50,"Truck":40}

base_time = (total_distance / vehicle_speed[vehicle]) * 60 if total_distance else 0
eta = base_time * traffic_factor[weather]

# ---------------- DASHBOARD ----------------
c1,c2,c3 = st.columns(3)
c1.metric("Distance (km)", round(total_distance,2))
c2.metric("ETA (min)", int(eta))
c3.metric("Stops", len(delivery_points))

st.info(f"🚗 Speed: {vehicle_speed[vehicle]} km/h | 🌦 Factor: {traffic_factor[weather]}")

# ---------------- LOAD DISPLAY ----------------
st.subheader("🚚 Vehicle Load Summary")
for v_idx, route in enumerate(routes):
    load = sum([demands[i-1] for i in route if i != 0])
    st.write(f"Vehicle {v_idx+1}: {load}/{vehicle_capacity}")

# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
if st.button("🚀 Predict"):

    if total_distance == 0:
        st.warning("Enter valid route")
    else:
        # Step 1: Create empty dataframe with ALL training columns
        df = pd.DataFrame(columns=columns)

        # Step 2: Fill ONLY safe numeric features
        df.loc[0, "latitude"] = pickup_lat
        df.loc[0, "longitude"] = pickup_lon
        df.loc[0, "distance_km"] = total_distance

        # Step 3: Handle weather safely
        weather_col = f"weather_condition_{weather}"
        if weather_col in columns:
            df.loc[0, weather_col] = 1

        # Step 4: Fill remaining NaNs with 0
        df = df.fillna(0)

        # ✅ CRITICAL FIX: Ensure column order EXACTLY matches
        df = df[columns]

        # Prediction
        pred_class = model.predict(df)[0]

        pred_minutes = int(eta + np.random.randint(-5,10))

        st.session_state.prediction = (pred_class, pred_minutes)
# ---------------- ANALYTICS ----------------
st.subheader("📊 Analytics")

history = pd.DataFrame({
    "weather": np.random.choice(["clear","rainy","foggy"], 50),
    "delay": np.random.randint(-5, 20, 50)
})

success_rate = (history["delay"] <= 0).mean()*100
st.metric("Success Rate (%)", round(success_rate,2))

avg_delay = history.groupby("weather")["delay"].mean()
st.bar_chart(avg_delay)