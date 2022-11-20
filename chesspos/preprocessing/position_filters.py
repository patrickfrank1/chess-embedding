import numpy as np


def subsample_opening_positions(ply: int) -> bool:
	selection_probability = min(ply / 40, 1.0)
	return np.random.random() < selection_probability

def heavy_subsampling(ply: int):
	selection_probability = subsample_opening_positions(ply) / 50.0
	return np.random.random() < selection_probability