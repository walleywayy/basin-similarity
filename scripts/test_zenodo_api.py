#!/usr/bin/env python3
"""Test script to verify Zenodo API access and file structure."""

import requests
import json

def test_zenodo_api():
    """Test Zenodo API access."""
    print("Testing Zenodo API access...")
    
    url = "https://zenodo.org/api/records/7540792/files"
    response = requests.get(url)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response keys: {list(data.keys())}")
        
        if 'entries' in data:
            print(f"Number of files: {len(data['entries'])}")
            
            for file_info in data['entries']:
                print(f"File: {file_info.get('key', 'Unknown')}")
                print(f"  Size: {file_info.get('size', 0) / (1024**3):.2f} GB")
                print(f"  Download URL: {file_info.get('links', {}).get('download', 'N/A')}")
                print()
        
        return True
    else:
        print(f"Error: {response.text}")
        return False

if __name__ == "__main__":
    test_zenodo_api()