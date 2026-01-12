import time
from collections import deque
from .interfaces import BaseModule

class PerformanceMonitor(BaseModule):
    def __init__(self):
        super().__init__("Telemetry")
        self.times = deque(maxlen=30)
        self.start_time = 0

    def initialize(self):
        self.start_time = time.time()
        return True

    def tick_start(self):
        self.step_start = time.time()

    def tick_end(self):
        latency = time.time() - self.step_start
        self.times.append(latency)

    def process(self, _=None):
        avg_latency = sum(self.times) / len(self.times) if self.times else 0
        fps = 1.0 / avg_latency if avg_latency > 0 else 0
        return {"fps": fps, "latency_ms": avg_latency * 1000}

    def shutdown(self):
        pass
