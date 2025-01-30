import requests
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=Path)
p = parser.parse_args()

url = 'http://127.0.0.1:5000/predict'
files = {'image': open(p.image_path, 'rb')}
response = requests.post(url, files=files)

print(response.json())