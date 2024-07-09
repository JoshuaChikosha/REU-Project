import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Field:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.sensors = []
        self.obstacles = []
        self.targets = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def add_target(self, target):
        self.targets.append(target)

    def visualize(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        for sensor in self.sensors:
            sensor_circle = patches.Circle(sensor.position, sensor.sensing_range, fill=False, edgecolor='blue', label='Sensing Range')
            ax.add_patch(sensor_circle)
            comm_circle = patches.Circle(sensor.position, sensor.communication_range, fill=False, edgecolor='red', linestyle='--', label='Communication Range')
            ax.add_patch(comm_circle)
            ax.plot(*sensor.position, 'bo')  # Sensor position

        for obstacle in self.obstacles:
            rect = patches.Rectangle(obstacle.position, obstacle.size, obstacle.size, linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)

        for target in self.targets:
            ax.plot(*target, 'rx')  # Target position

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Wireless Sensor Network Visualization')
        plt.grid(True)
        plt.show()
