import random
import numpy as np


# Set the initial seed for reproducibility
INITIAL_SEED = 422312


def generate_and_write_seeds(file_path, num_seeds, seed=INITIAL_SEED):
    rnd = random.Random(seed)

    # Write seeds to a text file
    with open(file_path, "w") as file:
        for _ in range(num_seeds):
            file.write(str(rnd.getrandbits(32)) + "\n")

    print("Seeds have been written to", file_path)


def read_file(file_path):
    with open(file_path, "r") as file:
        return [int(seed.strip()) for seed in file.readlines()]

def main_seeds():
    # Generate and write seeds for seeds_data.txt
    generate_and_write_seeds("seeds_data.txt", 100, seed=INITIAL_SEED)
    generate_and_write_seeds("seeds_training.txt", 700, seed=INITIAL_SEED)


if __name__ == '__main__':
    main_seeds()