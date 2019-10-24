import pandas as pd

from preprocessing import *
from train import *

TRAINING_FILE = "../data/train_32x32.mat"
TEST_FILE = "../data/test_32x32.mat"
TRAINING_DIR = ""
TEST_DIR = ""

def main():
    training_data = load(TRAINING_FILE)

if __name__ == "__main__":
    main()
