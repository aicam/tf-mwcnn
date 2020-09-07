import numpy as np
from max_sum import *


def corner_matrix_process(corner_matrix):
    rows, cols = np.where(corner_matrix == 1)
    difference_matrix = np.array([[rows[i], cols[i]] for i in range(len(rows))])
    difference_dict = []
    for i in range(len(difference_matrix)):
        difference_dict.append({'cell': difference_matrix[i], 'sum': np.sum(difference_matrix[i])})
    difference_dict = sorted(difference_dict, key=lambda x: x['sum'])
    variation_dict = []
    for i in range(len(difference_dict) - 1):
        variation_dict.append(
            {'cell': difference_dict[i]['cell'], 'diff': difference_dict[i + 1]['sum'] - difference_dict[i]['sum']})
    first, last = get_bounds([it['diff'] for it in variation_dict])
    cells = np.array([item['cell'].tolist() for item in variation_dict])[first: last]
    x_start, y_start = np.min(cells, axis=0)
    x_end, y_end = np.max(cells, axis=0)
    start_point = (y_start, x_start)
    end_point = (y_end, x_end)
    return cells, start_point, end_point
