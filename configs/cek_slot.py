import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(base_dir, "slots.json")

with open(file_path, "r") as f:
    slots = json.load(f)

print(f"Jumlah slot: {len(slots)}")