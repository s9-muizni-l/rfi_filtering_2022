import json

import numpy as np
from scipy.ndimage import uniform_filter1d

with open("train_prepare_params.json") as f:
    j = json.load(f)
    X_START = j["x-start"]
    X_END = j["x-end"]
    POINTS_TOTAL = j["points-total"]
    Y_AXIS = j["y-average"]
    NOISE_RANGE = (j["noise-range"]["min"], j["noise-range"]["max"])
    NOISE_FILTER = j["noise-size"]

    # Each indent contains: [start_val, center_val, end_val, set_max_val, start_exp, end_exp]
    INDENT = []
    indents_inp = j["indents"]
    for e in indents_inp:
        INDENT.append([
            e["start_val"],
            e["center_val"],
            e["end_val"],
            e["set_max_val"],
            e["start_exp"],
            e["end_exp"]
        ])

    INDENT = [
        [379, 385, 391, 3, 2, 2],
        [404, 414, 422, 1.5, 2, 2]
    ]
    INDENT_NOISE_RANGE = (j["indent-noise-range"]["min"], j["indent-noise-range"]["max"])
    INDENT_NOISE_FILTER = j["indent-noise-size"]

    R_COUNT = j["r-count"]


# Adding uniform noise
"""
Function applies uniform noise to the input
:input input_y: Array of elements to apply noise to
:input noise_range: List of min and max of applied noise
:input noise_filter: uniform_filter1d noise filter size
:return: Array of resulting values with added noise.
"""
def noisy(input_y, noise_range, noise_filter):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=len(input_y))
    return input_y + uniform_filter1d(noise, size=noise_filter)


"""
Function generates array of exponentially increasing/decreasing points.
:input start: starting value
:input end: value to end at
:input power: exponential power to use
:input num_points: points between start and end
:return: Array of exponentially increasing/decreasing points.
"""
def exp_space(start, end, power, num_points):
    start = np.power(start, 1 / float(power))
    end = np.power(end, 1 / float(power))
    return np.power(np.linspace(start, end, num_points), power)

"""
Applies indents to the input array
:input y_rand: input array
:return: Array of elements with included indents.
"""
def apply_indents(y_rand):
    for ind in INDENT:
        start_val = ind[0]
        center_val = ind[1]
        end_val = ind[2]
        set_max_val = ind[3]
        start_base = ind[4]
        end_base = ind[5]

        # Setting center value
        center_x = X_START - 9999
        for i in range(POINTS_TOTAL):
            if i == 0:
                continue

            if x[i - 1] < center_val < x[i]:
                y_rand[i] = set_max_val
                center_x = i
                break

        # Checking if such center exists
        if center_x >= X_START:
            i_start = False
            i_end = False

            for i in range(POINTS_TOTAL):
                if x[i] > start_val:
                    if not i_start:
                        i_start = True
                        diff = 0 - y_rand[i]

                        start_arr = exp_space(y_rand[i] + diff, y_rand[center_x] + diff, start_base, (center_x - i))
                        start_arr = noisy(start_arr - diff, NOISE_RANGE, NOISE_FILTER)

                        j = i
                        for e in start_arr:
                            y_rand[j] = e
                            j += 1

                if x[i] > end_val:
                    if not i_end:
                        i_end = True
                        diff = 0 - y_rand[i]

                        end_arr = exp_space(y_rand[center_x] + diff, y_rand[i] + diff, end_base, (i - 1 - center_x))
                        end_arr = noisy(end_arr - diff, INDENT_NOISE_RANGE, INDENT_NOISE_FILTER)

                        j = center_x
                        for e in end_arr:
                            y_rand[j] = e
                            j += 1
    return y_rand


# Creating an array of x values
x = np.linspace(X_START, X_END, POINTS_TOTAL)


for i in range(R_COUNT):
    y = np.full(POINTS_TOTAL, Y_AXIS)

    # Applying noise to the base values
    y_rand = noisy(y, NOISE_RANGE, NOISE_FILTER)
    # Changing noise to indent noise
    NOISE_RANGE = INDENT_NOISE_RANGE
    NOISE_FILTER = INDENT_NOISE_FILTER
    y_rand = apply_indents(y_rand)
    # Resetting noise
    NOISE_RANGE = (0, 0)
    y = apply_indents(y)

    # Saving results
    with open("parsed" + str(i+1) + ".npy", "wb") as file:
        np.save(file, y)

    with open("orig" + str(i+1) + ".npy", "wb") as file:
        np.save(file, y_rand)
