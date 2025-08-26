#!/usr/bin/env python3

import requests
import json

def test_bootstrap():
    print("Testing FPL Bootstrap API...")
    
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ API connection successful")
        print(f"Available keys: {list(data.keys())}")
        
        if 'teams' in data:
            print(f"Teams: {len(data['teams'])}")
        if 'elements' in data:
            print(f"Players: {len(data['elements'])}")
        if 'events' in data:
            print(f"Events: {len(data['events'])}")
        if 'fixtures' in data:
            print(f"Fixtures: {len(data['fixtures'])}")
        
        # Check if fixtures are in events
        if 'events' in data and len(data['events']) > 0:
            print(f"First event keys: {list(data['events'][0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_bootstrap()
