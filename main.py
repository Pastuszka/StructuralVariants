import numpy as np
import pysam
from scipy.stats import norm

def scan_chromosome(name, samfile, bin_length, chromosome_length):
    start = 0
    stop = start + bin_length
    counts = []
    starts = []
    gc_percents = []
    while start < chromosome_length:
        iter = samfile.fetch('chr21', start=start, stop=start + bin_length)
        start = start + bin_length
        i = 0
        total_bases = 0
        gc_bases = 0
        for elem in iter:
            i += 1
            seq = elem.query_sequence
            total_bases += len(seq)
            gc_bases += len([x for x in seq if x == 'C' or x == 'G'])
        if total_bases == 0:
            gc_percent = 0
        else:
            gc_percent = int(gc_bases / total_bases * 100)
        counts.append(i)
        starts.append(start)
        gc_percents.append(gc_percent)
        prt += 1
        if prt == 1000:
            progress = int((start / chromosome_length) * 100)
            print(str(progress) + '%', end='\r')
            prt = 0

    return np.array(counts), np.array(starts), np.array(gc_percents)


def gc_correction(counts, gc_percents):
    unique_gc = np.unique(gc_percents)
    rd_gcs = {}
    for elem in unique_gc:
        rd = counts[(gc_percents == elem) & (counts != 0)]
        rd_gcs[elem] = np.mean(rd)
    rd_global = np.mean(counts)
    rd_gcs_vec = [rd_gcs[i] for i in gc_percents]
    rd_corrected = (rd_global / rd_gcs_vec) * counts
    return rd_corrected, rd_global


def mean_shift_correction(counts, rd_corrected, rd_global):
    _, h0 = norm.fit(rd_corrected)
    hr = [np.sqrt(counts[i] / rd_global) * h0 if rd_corrected[i] > rd_global / 4 else h0 / 2 for i in
          range(len(rd_corrected))]


def smooth_signal(rd, ms):
    start = 0
    rd_new = np.empty_like(rd)
    for i in range(len(ms)):
        if (i == len(ms) - 1) or (ms[i] <= 0 < ms[i + 1]):
            rd_new[start:i+1] = np.mean(rd[start:i+1])
            start = i+1
    return rd_new
