import numpy as np
from sensor import Sensor
from tabu import TabuSearch

class Application:
    def __init__(self, required_coverage, k, sensing_range, communication_range, num_sensors):
        self.required_coverage = required_coverage
        self.k = k
        self.sensing_range = sensing_range
        self.communication_range = communication_range
        self.num_sensors = num_sensors

    def run_tabu_search(self, field):
        # Initialize TabuSearch with max_iterations set to 1000
        ts = TabuSearch(field, self.num_sensors, self.communication_range, self.sensing_range, field.targets, self.k, max_iterations=400)
        best_solution = ts.run()
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

    def count_coverage(self, target, sensors):
        count = 0
        for sensor in sensors:
            if sensor.is_within_coverage(target[0], target[1]):
                count += 1
        return count

    def run_ga(self, simulation_field, generations=1000, population_size=50):
        population = self.initialize_population(simulation_field, population_size)
        best_fitness = 0
        best_coverage = 0
        print("Initial Population Fitness:")
        for i, individual in enumerate(population):
            fit, covered_targets = self.fitness(simulation_field, individual)
            print(
                f"Individual {i}: Fitness = {fit:.4f}, Covered Targets = {covered_targets}, Positions = {[sensor.position for sensor in individual]}")

        for gen in range(generations):
            new_population = []
            for individual in population:
                new_individual = self.perform_ga_operations(simulation_field, individual, population)
                new_population.append(new_individual)
            population = self.select_best_individuals(new_population, simulation_field)

            while len(population) < population_size:
                indices = np.random.choice(len(population), 2, replace=False)
                parent1, parent2 = population[indices[0]], population[indices[1]]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1, simulation_field)
                offspring2 = self.mutate(offspring2, simulation_field)
                population.append(offspring1)
                population.append(offspring2)

            # Extract fitness and coverage information
            fitness_coverage_pairs = [self.fitness(simulation_field, individual) for individual in population]
            current_best_fitness, current_best_coverage = max(fitness_coverage_pairs, key=lambda x: x[0])

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_coverage = current_best_coverage

            print(
                f"Generation {gen + 1}/{generations}, best fitness: {best_fitness:.4f}, best coverage: {best_coverage}/{len(simulation_field.targets)}")
            for i, (fit, covered_targets) in enumerate(fitness_coverage_pairs):
                print(
                    f"Individual {i}: Fitness = {fit:.4f}, Covered Targets = {covered_targets}, Positions = {[sensor.position for sensor in population[i]]}")

        # Find the best solution
        best_solution = max(population, key=lambda x: self.fitness(simulation_field, x)[0])
        simulation_field.sensors = best_solution
        return best_solution

    def initialize_population(self, field, population_size):
        population = []
        for _ in range(population_size):
            individual = [
                Sensor(
                    (np.random.uniform(0, field.width), np.random.uniform(0, field.height)),
                    self.communication_range,
                    self.sensing_range
                ) for _ in range(self.num_sensors)
            ]
            population.append(individual)
        return population

    def perform_ga_operations(self, field, individual, population):
        new_individual = individual.copy()
        if np.random.rand() < 0.7:  # Crossover probability
            parent2 = population[np.random.randint(len(population))]
            new_individual, _ = self.crossover(new_individual, parent2)
        new_individual = self.mutate(new_individual, field)
        return new_individual

    def select_best_individuals(self, population, field):
        sorted_population = sorted(population, key=lambda x: self.fitness(field, x), reverse=True)
        best_individuals = sorted_population[:len(population) // 2]
        while len(best_individuals) < len(population):
            best_individuals.append(self.initialize_population(field, 1)[0])
        return best_individuals

    def crossover(self, parent1, parent2):
        crossover_point = len(parent1) // 2
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring1 = offspring1[:self.num_sensors]
        offspring2 = offspring2[:self.num_sensors]
        print(f"Crossover: Offspring1 = {offspring1}, Offspring2 = {offspring2}")
        return offspring1, offspring2

    def mutate(self, individual, field):
        mutation_rate = 0.5  # Increase the mutation rate
        for sensor in individual:
            if np.random.rand() < mutation_rate:
                new_position = (
                    sensor.position[0] + np.random.uniform(-10, 10),
                    sensor.position[1] + np.random.uniform(-10, 10)
                )
                sensor.position = (
                    min(max(new_position[0], 0), field.width),
                    min(max(new_position[1], 0), field.height)
                )
        return individual