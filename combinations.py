#!/usr/bin/python3
"""
File:       combinations.py
Author:     Martijn E.N.F.L. Schendstok
Date:       June 2020
"""
import itertools
import sys


def get_combinations():
    """
    Gets all spectral feature combinations, in order.
    :return com:    list(list())
    """
    #dict = {"A": "Max frequency", "B": "Min frequency", "C": "Bandwidth", "D": "Centroid", "E": "MFCC", "F": "Chroma",
    #        "G": "Zero crossing rate", "H": "RMS"}
    dict = {"A": "spectral_rolloff_max", "B": "spectral_rolloff_min", "C": "spectral_bandwidth",
            "D": "spectral_centroid", "E": "mfcc", "F": "chroma_stft", "G": "zero_crossing_rate", "H": "rms"}
    # lst = ["Max frequency", "Min frequency", "Bandwidth", "Centroid", "MFCC", "Chroma", "Zero crossing rate", "RMS"]
    lst = ["A", "B", "C", "D", "E", "F", "G", "H"]
    combs = []

    for i in range(1, len(lst) + 1):
        els = [list(x) for x in itertools.combinations(lst, i)]
        combs.extend(els)

    comb = list()
    for c in combs:
        #if len(c) == 1: comb.append(",".join(c))  # All features seperate
        #if len(c) > 1: comb.append(",".join(c))  # All combinations
        comb.append(",".join(c))  # Everything
    comb.sort()

    com = list()
    for c in comb:
        x = list()
        c_lst = c.split(",")
        for c2 in c_lst:
            x.append(dict[c2])
        com.append(x)

    return com


def main(argv):
    com = get_combinations()
    print(com)
    for c in com:
        print(", ".join(c))


if __name__ == "__main__":
    main(sys.argv)
