import numpy as np
from sensor import Sensor
from alg import Alg

class Application:
    def __init__(self, required_coverage, k, sensing_range, communication_range, num_sensors):
        self.required_coverage = required_coverage
        self.k = k
        self.sensing_range = sensing_range
        self.communication_range = communication_range
        self.num_sensors = num_sensors

    def run_alg(self, field):
        # Initialize with max_iterations set to 1000
        a = Alg(field, self.num_sensors, self.communication_range, self.sensing_range, field.targets, self.k, max_iterations=5000)
        best_solution = a.run()
        return best_solution


    # Other methods remain unchanged

    def fitness(self, field, sensors):
        covered_targets = 0
        for target in field.targets:
            if self.count_coverage(target, sensors) >= self.k:
                covered_targets += 1

        coverage_ratio = covered_targets / len(field.targets)

        # Connectivity incentive
        connected_sensors = 0
        total_pairs = 0
        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                total_pairs += 1
                if sensors[i].is_within_range(sensors[j]):
                    connected_sensors += 1

        connectivity_ratio = connected_sensors / total_pairs if total_pairs > 0 else 0

        # Weighted sum of coverage and connectivity
        fitness_score = 1.0 * coverage_ratio + 0.2 * connectivity_ratio  # Adjust weights as needed
        return fitness_score, covered_targets

    def total_distance_to_target(self, target, sensors):
        return sum(((sensor.position[0] - target[0]) ** 2 + (sensor.position[1] - target[1]) ** 2) ** 0.5 for sensor in
                   sensors)

 