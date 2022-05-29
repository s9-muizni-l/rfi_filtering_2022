import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
from baseband import vdif

with open("process_params.json") as f:
    j = json.load(f)
    VDIF_FILE = j["vdif-file"]
    SAMPLE_RATE = j["sample-rate"]
    DURATION = j["duration"]

fh = vdif.open(VDIF_FILE, 'rb')
fs = fh.read_frameset()

for i in range(1):
    y = []
    fr = fh.read_frame()
    converted = fr.data

    yfin = []
    for n in range(len(converted[0])):
        yfin.append([])

    for e in range(len(converted)):
        for n in range(len(converted[e])):
            yfin[n].append(converted[e][n])

    for i in range(16):
        y = yfin[i]

        N = int(SAMPLE_RATE * DURATION)
        yf = rfft(y)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)

        plt.plot(xf[1:], np.abs(yf[1:]))
        plt.show()

fh.close()


# -1 00
# katrs kanāls ir 8mhz
# tie 16 kanāli ir takā ar nobīdi, skatīt attēlu (pirmie 8 vienā pusē, otri 8 otrā pusē)
# pārbaudīt ar reverso inženieriju kā tās vērtības tiek saglabātas
# 2 bit sample nozīmē, ka katra vērtība ar 2 bitiem ir atzīmēta. t.i. kopumā ir 00 01 10 11 vērtības, jāatrod
# kura atbilst kurai -1 => 00? piemēram.
