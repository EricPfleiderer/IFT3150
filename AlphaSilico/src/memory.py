"""
Used to store training memory.
"""

# Standard
from collections import deque

# Local
from AlphaSilico.src import config


class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.ltmemory = deque(maxlen=memory_size)
        self.stmemory = deque(maxlen=memory_size)

    def commit_stmemory(self, params, y, pi):
        self.stmemory.append({'params': params,
                              'y': y,
                              'pi': pi,
                              })

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)
