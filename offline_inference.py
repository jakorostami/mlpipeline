import requests
import json
import pandas as pd


# Run this file in your terminal with `python offline_inference.py`
# Otherwise you can also run the Jupyter notebook in /notebooks/offline-inference.ipynb

input_data = {
    "Date": ["14/11/2013", "20/12/2013", "01/01/2014"],
    "Product": ["Shirt", "Hair Band", "Shirt"],
    "Gender": ["Male", "Female", "Female"],
    "Device_Type": ["Mobile", "Web", "Mobile"],
    "State": ["New York", "Washington", "New York"],
    "City": ["New York City", "Seattle", "New York City"],
    "Category": ["Clothing", "Accessories", "Clothing"],
    "Customer_Login_type": ["Member", "Guest", "Member"],
    "Delivery_Type": ["Normal Delivery", "one-day deliver", "one-day deliver"],
    "Individual_Price_US$": [10.0, 27, 10],
    "Time": ["22:44:41", "00:41:24", "15:20:33"],
    "Quantity": [500, 50, 100]
    }

api_url = "http://localhost:8000/predict"

print("Requesting API data")
response = requests.post(api_url, json.dumps(input_data))

print(response.json())



