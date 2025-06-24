class BaseAgent:
    def __init__(self, name):
        self.name = name

    def generate_signal(self, data):
        raise NotImplementedError
