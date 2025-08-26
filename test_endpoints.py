#!/usr/bin/env python3

import requests
import json

def test_endpoints():
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints...")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health: {response.status_code}")
    except Exception as e:
        print(f"❌ Health: {e}")
    
    # Test predictions
    try:
        response = requests.get(f"{base_url}/predictions?top_n=5")
        print(f"✅ Predictions: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Found {data.get('total_players', 0)} predictions")
    except Exception as e:
        print(f"❌ Predictions: {e}")
    
    # Test best11
    try:
        response = requests.get(f"{base_url}/best11?budget=100&formation=4-4-2")
        print(f"✅ Best11: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Team: {data.get('formation', 'N/A')}, Cost: {data.get('total_cost', 0)}")
        elif response.status_code == 404:
            print("   ❌ Best11 endpoint not found")
    except Exception as e:
        print(f"❌ Best11: {e}")
    
    # Test optimize-team
    try:
        response = requests.get(f"{base_url}/optimize-team?budget=100&formation=4-4-2")
        print(f"✅ Optimize-team: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Team: {data.get('formation', 'N/A')}, Cost: {data.get('total_cost', 0)}")
        elif response.status_code == 404:
            print("   ❌ Optimize-team endpoint not found")
    except Exception as e:
        print(f"❌ Optimize-team: {e}")

if __name__ == "__main__":
    test_endpoints()
