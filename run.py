import numpy as np
from application import Application
from field import Field
from sensor import Sensor
import matplotlib.pyplot as plt

# Parameters
sensing_range = 10
communication_range =  20
num_sensors = 25
num_targets = 5
field_width = 100
field_height = 100
num_simulations = 100  # Number of bootstrap simulations

# Create the field
field = Field(field_width, field_height)

# Randomly deploy targets
np.random.seed()  # For reproducibility, you can remove or change the seed for different random results
field.targets = [(np.random.uniform(0, field_width), np.random.uniform(0, field_height)) for _ in range(num_targets)]

# Randomly create sensors
sensors = [Sensor((np.random.uniform(0, field_width), np.random.uniform(0, field_height)), communication_range, sensing_range) for _ in range(num_sensors)]

# Add sensors to the field
for sensor in sensors:
    field.add_sensor(sensor)

# Visualize initial state of the field
print("Initial state of the field:")
field.visualize()

# Create the application and run the algorithm
app = Application(required_coverage=1, k=3, sensing_range=sensing_range, communication_range=communication_range, num_sensors=num_sensors)
best_solution = app.run_alg(field)

# Visualize the final state of the field
print("Final state of the field after optimization:")
field.sensors = best_solution
field.visualize()

"""
# Function to run multiple simulations
def run_simulations():
    results = []
    for sim in range(num_simulations):
        np.random.seed(sim)  # Set seed for reproducibility
        field = Field(field_width, field_height)
        field.targets = [(np.random.uniform(0, field_width), np.random.uniform(0, field_height)) for _ in
                         range(num_targets)]
        sensors = [Sensor((np.random.uniform(0, field_width), np.random.uniform(0, field_height)), communication_range,
                          sensing_range) for _ in range(num_sensors)]
        for sensor in sensors:
            field.add_sensor(sensor)

        app = Application(required_coverage=1, k=2, sensing_range=sensing_range,
                          communication_range=communication_range, num_sensors=num_sensors)
        best_solution = app.run_tabu_search(field)
        field.sensors = best_solution
        coverage, connectivity = app.fitness(field, best_solution)
        results.append((coverage, connectivity))

    return results


# Run simulations
simulation_results = run_simulations()

# Plotting results
coverages, connectivities = zip(*simulation_results)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(coverages, bins=10, color='blue', alpha=0.7)
plt.title('Coverage Distribution')
plt.xlabel('Coverage')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(connectivities, bins=10, color='green', alpha=0.7)
plt.title('Connectivity Distribution')
plt.xlabel('Connectivity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
"""