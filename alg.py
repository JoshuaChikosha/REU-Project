import numpy as np
from sensor import Sensor
import random
 
class Alg:
    def __init__(self, field, num_sensors, communication_range, sensing_range, targets, k, max_iterations=100):
        self.field = field
        self.num_sensors = num_sensors
        self.communication_range = communication_range
        self.sensing_range = sensing_range
        self.targets = targets
        self.k = k
        self.max_iterations = max_iterations
        self.population_size = 20  # Population size
        self.population = [self.initial_solution() for _ in range(self.population_size)]
        self.stagnation_counter = 0  # Counter to track stagnation
        self.temperature = 100  # Initial temperature for Simulated Annealing
        self.mutation_rate = 0.4  # Initial mutation rate
        self.elitism_rate = 0.1  # Elitism rate
        self.elite_solution = None  # Store the best solution

    def initial_solution(self):
        sensors = [Sensor(
            (np.random.uniform(0, self.field.width), np.random.uniform(0, self.field.height)),
            self.communication_range,
            self.sensing_range
        ) for _ in range(self.num_sensors)]
        return sensors

    def evaluate(self, sensors):
        # Calculate Coverage Ratio
        coverage_score = 0
        for target in self.targets:
            covered_count = sum(1 for sensor in sensors if ((sensor.position[0] - target[0]) ** 2 + (
                        sensor.position[1] - target[1]) ** 2) ** 0.5 <= sensor.sensing_range)
            coverage_score += min(covered_count, self.k) / self.k

        coverage_ratio = coverage_score / len(self.targets)

        # Calculate Connectivity Ratio
        connected_pairs = 0
        total_possible_pairs = len(sensors) * (len(sensors) - 1) / 2
        for i in range(len(sensors)):
            for j in range(i + 1, len(sensors)):
                if ((sensors[i].position[0] - sensors[j].position[0]) ** 2 + (
                        sensors[i].position[1] - sensors[j].position[1]) ** 2) ** 0.5 <= sensors[i].communication_range:
                    connected_pairs += 1
        connectivity_ratio = connected_pairs / total_possible_pairs if total_possible_pairs > 0 else 0

        # Adjust weights to favor coverage more
        if coverage_ratio < 1.0:
            return coverage_ratio  # Prioritize maximizing coverage first

        fitness_score = 0.9 * coverage_ratio + 0.1 * connectivity_ratio
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
                if np.random.rand() < self.mutation_rate:  # Mutation probability
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

    def accept_solution(self, current_fitness, new_fitness):
        """ Simulated Annealing acceptance criteria. """
        if new_fitness > current_fitness:
            return True
        else:
            delta = new_fitness - current_fitness
            acceptance_probability = np.exp(delta / self.temperature)
            return acceptance_probability > np.random.rand()

    def run(self):
        current_best = max(self.population, key=lambda x: self.evaluate(x))
        best_fitness = self.evaluate(current_best)
        self.elite_solution = current_best
        self.elite_fitness = best_fitness

        for iteration in range(self.max_iterations):
            new_population = []
            best, second_best = self.tournament_selection()
            
            # Elitism: retain the best individuals
            num_elites = int(self.elitism_rate * self.population_size)
            elites = sorted(self.population, key=lambda x: self.evaluate(x), reverse=True)[:num_elites]
            new_population.extend(elites)

            for individual in self.population[num_elites:]:
                new_individual = self.mutate_and_crossover(individual, best, second_best)
                current_fitness = self.evaluate(individual)
                new_fitness = self.evaluate(new_individual)
                
                if self.accept_solution(current_fitness, new_fitness):
                    new_population.append(new_individual)
                else:
                    new_population.append(individual)

            self.population = new_population
            current_best = max(self.population, key=lambda x: self.evaluate(x))
            current_fitness = self.evaluate(current_best)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                self.elite_solution = current_best
                self.elite_fitness = best_fitness
                self.stagnation_counter = 0  # Reset stagnation counter
                print(f"New Best Fitness at Iteration {iteration}: {best_fitness}")
            else:
                self.stagnation_counter += 1
                print(f"Iteration {iteration + 1}, Current Fitness: {current_fitness}, Best Fitness: {best_fitness}")

            # Dynamic parameter adjustment
            self.temperature *= 0.99  # Gradually decrease the temperature
            if self.stagnation_counter >= 10:  # If stagnation persists for 10 iterations
                print(f"Stagnation detected at iteration {iteration}. Applying perturbation.")
                self.guided_diversification()
                self.stagnation_counter = 0  # Reset counter after diversification

            # Adaptive mutation rate adjustment
            self.mutation_rate = 0.9 - (0.8 * iteration / self.max_iterations)

        return self.elite_solution

    def guided_diversification(self):
        """Guided diversification strategy to avoid premature restarts."""
        new_population = []
        perturbation_rate = 0.1  # Rate of perturbation

        for i in range(len(self.population)):
            if random.random() < 0.2:  # 20% chance to replace with a perturbed best solution
                best, second_best = self.tournament_selection()
                new_solution = self.mutate_and_crossover(best, best, second_best)
                new_population.append(new_solution)
            else:
                individual = self.population[i]
                perturbed_individual = []
                for sensor in individual:
                    if random.random() < perturbation_rate:
                        # Slightly perturb the position of the sensor
                        new_position = (
                            np.clip(sensor.position[0] + np.random.uniform(-0.1, 0.1) * self.field.width, 0, self.field.width),
                            np.clip(sensor.position[1] + np.random.uniform(-0.1, 0.1) * self.field.height, 0, self.field.height)
                        )
                    else:
                        new_position = sensor.position
                    perturbed_individual.append(Sensor(new_position, sensor.communication_range, sensor.sensing_range))
                new_population.append(perturbed_individual)

        self.population = new_population
