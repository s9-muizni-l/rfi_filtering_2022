import json
import math
import numpy as np
import concurrent.futures
from baseband import vdif

with open("vdif_params.json") as f:
    j = json.load(f)
    VDIF_FILE = j["vdif-file"]
    CHANNEL_AMOUNT = j["channel-amount"]
    FRAME_NUM = j["frame-num"]

    SPLIT_EVERY = j["split-every"]
    SAVE_FOLDER = j["save-folder"]
    SAVE_PREFIX = j["save-prefix"]

"""
Function reads a single vdif frame and stores it's content in a returning list
:param inp: Array of 2 elements - frame data index and data array itself
:return: List of each channel data array as Python list. First element of each sub-list is stored frame index.
"""
def read_frame(inp):
    data = inp[1]
    index = inp[0]
    top_size = len(data)

    y = []
    for i in range(CHANNEL_AMOUNT):
        y.append([])
        y[i].append(index)

    for n in range(top_size):
        for j in range(CHANNEL_AMOUNT):
            y[j].append(data[n][j])

    return y


saved_counter = 1

"""
Function saves provided list as a numpy file
:param ytemp: list to save
"""
def save_current(ytemp):
    global saved_counter
    yfin = []
    for i in range(CHANNEL_AMOUNT):
        yfin.append([])

    for i in range(CHANNEL_AMOUNT):
        for j in range(FRAME_NUM):
            yfin[i] += ytemp[i][j]

    for i in range(CHANNEL_AMOUNT):
        with open(SAVE_FOLDER + '/' + SAVE_PREFIX + str(saved_counter) + '-' + str(i + 1) + '.npy', 'wb') as file:
            np.save(file, np.asarray(yfin[i]))

    saved_counter += 1


if __name__ == '__main__':
    fh = vdif.open(VDIF_FILE, 'rb')
    fs = fh.read_frameset()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        frames = []
        n = 0

        # Store data frames in a python list, storing order as first element
        for i in range(FRAME_NUM):
            frames.append((n, fh.read_frame().data))
            n += 1
            if n == SPLIT_EVERY:
                n = 0

        parts = math.ceil(FRAME_NUM/SPLIT_EVERY)

        for split in range(parts):
            # Preparing array for data array contents
            ytemp = []
            for i in range(CHANNEL_AMOUNT):
                ytemp.append([])
                for j in range(FRAME_NUM):
                    ytemp[i].append([])

            # How many frames will be used
            start = SPLIT_EVERY * split
            end = SPLIT_EVERY + start
            if end > len(frames):
                end = len(frames)

            # How parallel results will be retrieved
            results = [executor.submit(read_frame, arg) for arg in frames[start:end]]

            # Runs in parallel
            for f in concurrent.futures.as_completed(results):
                res = f.result()

                for i in range(CHANNEL_AMOUNT):
                    ytemp[i][res[i][0]] = res[i][1:]

            # Save current "part" in a file
            save_current(ytemp)
            print(f'Saved part {split + 1} of {parts}')

    fh.close()
    print('Conversion finished')
