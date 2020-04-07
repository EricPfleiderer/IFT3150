"""
Settings related to model saving and loading.
"""

# Standard
import os

# Training initialization
INITIAL_RUN_NUMBER = 1
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None

# Directories, paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
run_folder = ROOT_DIR + '/src/models/run'
archive_folder = ROOT_DIR + '/src/models/archive'



