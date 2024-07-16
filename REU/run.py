import numpy as np
from application import Application
from field import Field
from sensor import Sensor
import matplotlib.pyplot as plt

# Parameters
sensing_range = 10
communication_range = 20
num_sensors = 35
field_width = 100
field_height = 100

# Run a single simulation
def run_single_simulation():
    np.random.seed(42)  # Set seed for reproducibility
    field = Field(field_width, field_height)
    # Add sensors to the field
    sensors = [Sensor((np.random.uniform(0, field_width), np.random.uniform(0, field_height)), communication_range, sensing_range) for _ in range(num_sensors)]
    for sensor in sensors:
        field.add_sensor(sensor)

    app = Application(required_coverage=1, k=2, sensing_range=sensing_range, communication_range=communication_range, num_sensors=num_sensors)
    best_solution = app.run_tabu_search(field)
    field.sensors = best_solution
    coverage, connectivity = app.fitness(field, best_solution)

    return coverage, connectivity, field

# Run the simulation and retrieve results
coverage, connectivity, field = run_single_simulation()

# Output results
print(f"Coverage: {coverage * 100:.2f}%")
print(f"Connectivity: {connectivity * 100:.2f}%")

# Visualize the field with sensors and coverage areas
field.visualize()
