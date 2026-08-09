import yaml

with open("config/cities.yaml", "r") as file:
    data = yaml.safe_load(file)

print(data)