import time
from contextlib import asynccontextmanager

class Timer:
    def __init__(self):
        self.start_time = None
        self.duration = None

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.perf_counter() - self.start_time
        # You can log it here or just let the caller access .duration
        print(f"Operation took {self.duration:.4f} seconds")