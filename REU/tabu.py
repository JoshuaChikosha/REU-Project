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
        # Create grid points for the field to simulate area coverage
        grid_points = [(x, y) for x in np.linspace(0, self.field.width, num=20)
                       # 20 can be adjusted based on desired granularity
                       for y in np.linspace(0, self.field.height, num=20)]

        covered_points = sum(
            1 for point in grid_points if any(sensor.is_within_coverage(point[0], point[1]) for sensor in sensors))
        area_coverage_ratio = covered_points / len(grid_points)

        # Calculate Connectivity Ratio
        connected_pairs = 0
        total_possible_pairs = len(sensors) * (len(sensors) - 1) / 2
        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                if ((sensors[i].position[0] - sensors[j].position[0]) ** 2 + (
                        sensors[i].position[1] - sensors[j].position[1]) ** 2) ** 0.5 <= sensors[i].communication_range:
                    connected_pairs += 1
        connectivity_ratio = connected_pairs / total_possible_pairs if total_possible_pairs > 0 else 0

        fitness_score =  area_coverage_ratio  # Adjust weights to favor coverage more

        return fitness_score

    def tournament_selection(self):
        # Selects the best and second-best individuals from a sample of the population
        sample = random.sample(self.population, 5)
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
        for sensor in individual:
            if self.is_isolated(sensor, individual):
                # Force move to a new position if isolated
                new_position = (np.random.uniform(0, self.field.width), np.random.uniform(0, self.field.height))
            else:
                if np.random.rand() < 0.7:  # Mutation probability
                    # Apply differential mutation
                    b_sensor = best[individual.index(sensor)]
                    s_sensor = second_best[individual.index(sensor)]
                    new_position = np.array(sensor.position) + F1 * (
                                np.array(b_sensor.position) - np.array(sensor.position)) + F2 * (
                                               np.array(s_sensor.position) - np.array(sensor.position))
                    new_position = np.clip(new_position, [0, 0], [self.field.width, self.field.height])
                else:
                    # Keep current position if no mutation
                    new_position = sensor.position
            mutated.append(Sensor(tuple(new_position), sensor.communication_range, sensor.sensing_range))
        return mutated

    def run(self):
        current_best = max(self.population, key=lambda x: self.evaluate(x))
        best_fitness = self.evaluate(current_best)

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
