import numpy as np
from sensor import Sensor
import random


class TabuSearch:
    def __init__(self, field, num_sensors, communication_range, sensing_range, targets, k, max_iterations=100, tabu_size=50):
        self.field = field
        self.num_sensors = num_sensors
        self.communication_range = communication_range
        self.sensing_range = sensing_range
        self.targets = targets
        self.k = k
        self.max_iterations = max_iterations
        self.tabu_list = []
        self.tabu_size = tabu_size
        self.population = [self.initial_solution() for _ in range(20)]  # Population size of 20

    def initial_solution(self):
        sensors = [Sensor(
            (np.random.uniform(0, self.field.width), np.random.uniform(0, self.field.height)),
            self.communication_range,
            self.sensing_range
        ) for _ in range(self.num_sensors)]
        return sensors

    def evaluate(self, sensors):
        # Calculate Coverage Ratio
        covered_targets = 0
        for target in self.targets:
            covered_count = sum(1 for sensor in sensors if ((sensor.position[0] - target[0]) ** 2 +
                                                            (sensor.position[1] - target[
                                                                1]) ** 2) ** 0.5 <= sensor.sensing_range)
            if covered_count >= self.k:
                covered_targets += 1
        coverage_ratio = covered_targets / len(self.targets)

        # Check if all targets are k-covered
        all_k_covered = covered_targets == len(self.targets)

        # Calculate Connectivity Ratio only if all targets are k-covered
        if all_k_covered:
            connected_pairs = 0
            sensor_connections = {sensor: 0 for sensor in sensors}
            for i in range(len(sensors)):
                for j in range(i + 1, len(sensors)):
                    if ((sensors[i].position[0] - sensors[j].position[0]) ** 2 +
                        (sensors[i].position[1] - sensors[j].position[1]) ** 2) ** 0.5 <= sensors[
                        i].communication_range:
                        sensor_connections[sensors[i]] += 1
                        sensor_connections[sensors[j]] += 1

            # Only count pairs where both sensors have at most m connections
            m = 2  # You can adjust this based on your requirement for m-connectivity
            connected_pairs = sum(connections <= m for connections in sensor_connections.values())

            total_possible_pairs = len(sensors) * (len(sensors) - 1) / 2
            connectivity_ratio = connected_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
        else:
            connectivity_ratio = 0  # Ignore connectivity if not all targets are k-covered

        # Calculate Penalty for Under-Covered Targets
        penalty = sum(1 for target in self.targets if sum(1 for sensor in sensors if (
                (sensor.position[0] - target[0]) ** 2 + (
                sensor.position[1] - target[1]) ** 2) ** 0.5 <= sensor.sensing_range) < self.k)

        # Adjust weights to favor coverage more
        fitness_score = 0.9 * coverage_ratio + 0.1 * connectivity_ratio - 0.4 * penalty
        return fitness_score

    def tournament_selection(self):
        # Selects the best and second-best individuals from a larger sample of the population
        sample = random.sample(self.population, 10)  # Increased sample size for better diversity
        sample_sorted = sorted(sample, key=lambda x: self.evaluate(x), reverse=True)
        return sample_sorted[0], sample_sorted[1]

    def is_isolated(self, sensor, sensors):
        """ Check if the sensor is isolated from other sensors. """
        return not any(np.sqrt(
            (sensor.position[0] - s.position[0]) ** 2 + (sensor.position[1] - s.position[1]) ** 2) <= self.sensing_range
                       for s in sensors if s != sensor)

    def mutate_and_crossover(self, individual, best, second_best):
        F1, F2 = 0.8, 0.5  # Scaling factors
        mutated = []
        # Determine coverage of each target
        coverage_count = {tuple(target): [] for target in self.field.targets}
        for sensor in individual:
            for target in self.field.targets:
                if ((sensor.position[0] - target[0]) ** 2 + (
                        sensor.position[1] - target[1]) ** 2) ** 0.5 <= sensor.sensing_range:
                    coverage_count[tuple(target)].append(sensor)

        # List of targets with insufficient coverage
        under_covered_targets = {target: sensors for target, sensors in coverage_count.items() if len(sensors) < 4}

        for idx, sensor in enumerate(individual):
            is_covering_under_covered_target = any(
                sensor in sensors for target, sensors in under_covered_targets.items())

            if not is_covering_under_covered_target and np.random.rand() < 0.9:  # Only move sensors not covering under-covered targets
                if under_covered_targets:
                    # Preferentially move towards the nearest under-covered target
                    distances = {tuple(target): np.linalg.norm(np.array(sensor.position) - np.array(target)) for target
                                 in under_covered_targets}
                    nearest_target = min(distances, key=distances.get)
                    direction = np.array(nearest_target) - np.array(sensor.position)
                    step_size = min(20, np.linalg.norm(direction))  # Dynamic step size
                    direction = direction / np.linalg.norm(direction) * step_size
                    new_position = np.array(sensor.position) + direction
                else:
                    # Random movement if no under-covered targets
                    direction = np.random.uniform(-20, 20, 2)
                    new_position = np.array(sensor.position) + direction

                new_position = tuple(np.clip(new_position, [0, 0], [self.field.width, self.field.height]))
            else:
                # Sensors covering under-covered targets do not move
                new_position = sensor.position

            mutated.append(Sensor(new_position, sensor.communication_range, sensor.sensing_range))

        return mutated

    def run(self):
        current_best = max(self.population, key=lambda x: self.evaluate(x))
        best_fitness = self.evaluate(current_best)
        print(f"Initial Best Fitness: {best_fitness}")

        for iteration in range(self.max_iterations):
            new_population = []
            best, second_best = self.tournament_selection()
            for individual in self.population:
                new_individual = self.mutate_and_crossover(individual, best, second_best)
                if self.evaluate(new_individual) > self.evaluate(individual):
                    new_population.append(new_individual)
                else:
                    new_population.append(individual)

            self.population = new_population
            current_best = max(self.population, key=lambda x: self.evaluate(x))
            current_fitness = self.evaluate(current_best)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                print(f"New Best Fitness at Iteration {iteration}: {best_fitness}")
            else:
                print(f"Iteration {iteration + 1}, Current Fitness: {current_fitness}, Best Fitness: {best_fitness}")

        return current_best

    def enforce_coverage_and_connectivity(self, sensor, sensors, field):
        # Check if the sensor covers any target
        is_covering_any_target = any(sensor.is_within_coverage(target[0], target[1]) for target in field.targets)

        if not is_covering_any_target:
            # Move sensor towards the nearest target with a larger step if it's not covering any targets
            nearest_target = min(field.targets, key=lambda t: np.sqrt(
                (sensor.position[0] - t[0]) ** 2 + (sensor.position[1] - t[1]) ** 2))
            sensor.position = self.move_towards_large_step(sensor.position, nearest_target)

    def move_towards_large_step(self, start_pos, end_pos):
        direction = np.array(end_pos) - np.array(start_pos)
        norm_direction = direction / np.linalg.norm(direction)
        new_position = np.array(start_pos) + norm_direction * 20  # Increase the movement step to 20 units
        return tuple(new_position)





    def is_overconnected(self, sensor, sensors):
        connected_sensors = sum(1 for s in sensors if s != sensor and sensor.is_within_range(s))
        return connected_sensors > 2  # Assuming m = 2 for m-connectivity

    def mutate(self, individual, field):
        # Iterate through each sensor in the individual (sensor array)
        for sensor in individual:
            # First, check if the sensor is already covering a k-covered target
            is_covering_k_targets = self.count_coverage(sensor.position, individual) >= self.k
            if not is_covering_k_targets:
                # Find sensors that are within communication range and covering a target
                communicable_sensors = [s for s in individual if s != sensor and sensor.is_within_range(s)]
                # Filter those sensors that are covering any target
                potential_helpers = [s for s in communicable_sensors if any(
                    s.is_within_coverage(t[0], t[1]) for t in field.targets if
                    self.count_coverage(t, individual) < self.k)]

                # If there are any potential helpers, move the current sensor towards a covered target by one of the helpers
                if potential_helpers:
                    # Select the first helper and move towards its covered target
                    helper = random.choice(potential_helpers)
                    covered_targets_by_helper = [t for t in field.targets if
                                                 helper.is_within_coverage(t[0], t[1]) and self.count_coverage(t,
                                                                                                               individual) < self.k]
                    if covered_targets_by_helper:
                        nearest_target = min(covered_targets_by_helper,
                                             key=lambda t: np.linalg.norm(np.array(sensor.position) - np.array(t)))
                        sensor.position = self.move_towards(sensor.position, nearest_target)

        return individual

    def count_coverage(self, target, sensors):
        """Count how many sensors are covering a specific target."""
        return sum(1 for sensor in sensors if sensor.is_within_coverage(target[0], target[1]))

    def move_towards(self, start_pos, end_pos):
        """Move the sensor towards the target by a fixed step."""
        direction = np.array(end_pos) - np.array(start_pos)
        norm_direction = direction / np.linalg.norm(direction)
        step_size = 5  # Define the step size for moving towards the target
        new_position = np.array(start_pos) + norm_direction * step_size
        return tuple(new_position)

