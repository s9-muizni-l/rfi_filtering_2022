import json
import os
import numpy as np
from scipy.ndimage import uniform_filter1d

with open("train_prepare_params.json") as f:
    j = json.load(f)
    X_FILE = j["x-file"]
    DATA_FOLDER = j["data-folder"]
    PREFIX = j["prefix"]

    THRESH_MAX = j["thresh-max"]
    THRESH_MIN = j["thresh-min"]
    POINTS_AROUND = j["points-around"]
    CENTER = j["center"]
    keep_inp = j["keep"]

    KEEP = []
    for e in keep_inp:
        KEEP.append([e["start"], e["end"], e["val"]])

# Getting indices to keep from X file
with open(X_FILE, "rb") as file:
    x_arr = np.load(file).tolist()

keep_final = [[-1, -1, KEEP[0][2]]]
center_final = -1

# Getting interval indices to keep
for i in range(len(x_arr)):
    val = float(x_arr[i])
    if round(val) == CENTER:
        center_final = i

    for keep_ind in range(len(KEEP)):
        if keep_ind > len(keep_final) - 1:
            keep_final.append([-1, -1, KEEP[keep_ind][2]])

        group = KEEP[keep_ind]

        if group[0] <= val <= group[1]:
            if keep_final[keep_ind][0] < 0:
                keep_final[keep_ind][0] = i
            else:
                keep_final[keep_ind][1] = i

# Raise exception if it is not possible to include selected interval
if center_final - POINTS_AROUND < 0:
    raise Exception(f"Cannot select interval with center {center_final} and {POINTS_AROUND} points around it (-)")
elif center_final + POINTS_AROUND > len(x_arr):
    raise Exception(f"Cannot select interval with center {center_final} and {POINTS_AROUND} points around it (+)")

min_ind = center_final - POINTS_AROUND
max_ind = center_final + POINTS_AROUND
result = []

# Reading data
for f in os.listdir(DATA_FOLDER):
    with open(DATA_FOLDER + "/" + f, "rb") as file:
        result.append(np.load(file).tolist())

tot = 1
for e in result:
    y = []

    # Appending only values in selected interval
    x = []
    for ind in range(len(e)):
        if min_ind <= ind <= max_ind:
            if THRESH_MIN < e[ind] < THRESH_MAX:
                y.append(e[ind])
            else:
                y.append(0)
            x.append(len(y))

    # Getting approx. fully de-noised line
    y2 = uniform_filter1d(y, size=1000)

    # Keeping original points from selected intervals
    yfin = []
    ind = 0
    for ind2 in range(len(e)):
        if min_ind <= ind2 <= max_ind:
            contains = False

            for group in keep_final:
                if group[0] <= ind2 <= group[1] and y[ind] > group[2]:
                    yfin.append(y[ind])
                    contains = True
                    break

            if not contains:
                yfin.append(y2[ind])

            ind += 1

    # Saving in parsed and original files
    with open("parsed_a2/" + PREFIX + str(tot) + ".npy", "wb") as file:
        np.save(file, np.asarray(yfin))

    with open("orig_a2/" + PREFIX + str(tot) + ".npy", "wb") as file:
        np.save(file, np.asarray(y))

    tot += 1
