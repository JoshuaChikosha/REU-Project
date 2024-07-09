import numpy as np
from application import Application
from field import Field
from sensor import Sensor

# Parameters
sensing_range = 10
communication_range = 20
num_sensors = 25
num_targets = 5
field_width = 100
field_height = 100

# Create the field
field = Field(field_width, field_height)

# Randomly deploy targets
np.random.rand()  # For reproducibility, you can remove or change the seed for different random results
field.targets = [(np.random.uniform(0, field_width), np.random.uniform(0, field_height)) for _ in range(num_targets)]

# Randomly create sensors
sensors = [Sensor((np.random.uniform(0, field_width), np.random.uniform(0, field_height)), communication_range, sensing_range) for _ in range(num_sensors)]

# Add sensors to the field
for sensor in sensors:
    field.add_sensor(sensor)

# Visualize initial state of the field
print("Initial state of the field:")
field.visualize()

# Create the application and run the GA algorithm
app = Application(required_coverage=1, k=2, sensing_range=sensing_range, communication_range=communication_range, num_sensors=num_sensors)
best_solution = app.run_ga(field, generations=5000, population_size=20)  # Adjusted for testing

# Visualize the final state of the field
print("Final state of the field after GA optimization:")
field.sensors = best_solution
field.visualize()