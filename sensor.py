# sensor.py
class Sensor:
    def __init__(self, position, communication_range, sensing_range):
        self.position = position
        self.communication_range = communication_range
        self.sensing_range = sensing_range

    def is_within_range(self, other_sensor):
        distance = ((self.position[0] - other_sensor.position[0]) ** 2 + (self.position[1] - other_sensor.position[1]) ** 2) ** 0.5
        return distance <= self.communication_range

    def is_within_coverage(self, x, y):
        return ((self.position[0] - x) ** 2 + (self.position[1] - y) ** 2) ** 0.5 <= self.sensing_range

