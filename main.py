import numpy as np
import pysam
from scipy.stats import norm
from scipy.stats import t

def get_chromosome_lengths(samfile, chromosomes):
    sqs = samfile.header['SQ']
    chromosome_lengths = {}
    for sq in sqs:
        name = sq['SN']
        length = sq['LN']
        if sq['SN'] in chromosomes:
            chromosome_lengths[name] = length
    return chromosome_lengths


def scan_chromosome(name, samfile, bin_length, chromosome_length):
    start = 0
    stop = start + bin_length
    counts = []
    starts = []
    gc_percents = []
    prt = 0
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

def freeze_segments(segments, rd_corrected, hr):
    average_rd = np.mean(rd_corrected)
    is_frozen = np.full(len(rd_corrected), False)
    frozen_segment_starts = []
    averages = np.empty_like(segments)
    sds = np.empty_like(segments)
    ns = np.empty_like(segments)
    segment_edges = segments
    segment_edges.append(len(rd_corrected))
    for i in range(segments):
        averages[i] = np.mean(rd_corrected[segment_edges[i]:segment_edges[i+1]])
        sds[i] = np.std(rd_corrected[segment_edges[i]:segment_edges[i+1]])
        ns[i] = segment_edges[i+1] - segment_edges[i]
    for i in range(segments):
        t1 = ((average_rd - averages[i]) / sds[i]) * np.sqrt(ns[i])
        p1 = 2*(t.cdf(-abs(t1), ns[i] - 1))
        cond1 = p1 < 0.05
        if i > 0:
            mean_diff = averages[i-1] - averages[i]
            t2 = (mean_diff / sds[i]) * np.sqrt(2)
            p2 = 2 * (t.cdf(-abs(t2), 1))
            cond2 = p2 < 0.01 or abs(mean_diff) >= 2 * hr[i]
        else:
            cond2 = True
        if i > 0:
            mean_diff = averages[i+1] - averages[i]
            t3 = (mean_diff / sds[i]) * np.sqrt(2)
            p3 = 2 * (t.cdf(-abs(t3), 1))
            cond3 = p3 < 0.01 or abs(mean_diff) >= 2 * hr[i]
        else:
            cond3 = True
        if cond1 and cond2 and cond3:
            frozen_segment_starts.append(segments[i])
            is_frozen[segment_edges[i]:segment_edges[i+1]] = True
    return is_frozen, frozen_segment_starts

def mean_shift(counts, rd_corrected, rd_global, limit = 128):
    _, h0 = norm.fit(rd_corrected)
    hr = [np.sqrt(counts[i] / rd_global) * h0 if
          rd_corrected[i] > rd_global/4 else
          h0/2 for i in range(len(rd_corrected))]
    hb = 2
    is_frozen = np.full(len(counts), False)
    frozen_segment_starts = []
    while hb < limit:
        segments = mean_shift_step(rd_corrected, hb, hr, is_frozen, frozen_segment_starts)
        is_frozen, frozen_segment_starts = freeze_segments(segments, rd_corrected, hr)
        hb += 1


def mean_shift_gradient(rd, hb, hr):
    totals = np.zeros_like(rd)
    j = np.arange(len(rd))
    for i in range(len(rd)):
        totals[i] = np.sum(
            (j - i) * np.exp(-((j - i) ** 2) / (2 * (hb ** 2))) * np.exp(-((rd[j] - rd[i]) ** 2) / (2 * (hr[i] ** 2))))
    return totals

def mean_shift_partition(rd, gradient):
    start = 0
    segment_starts = [0]
    rd_new = np.empty_like(rd)
    for i in range(len(gradient)):
        if i == len(gradient) - 1:
            rd_new[start:i + 1] = np.mean(rd[start:i + 1])
        if gradient[i] <= 0 < gradient[i + 1]:
            rd_new[start:i + 1] = np.mean(rd[start:i + 1])
            start = i + 1
            segment_starts.append(start)
    return segment_starts, rd_new

def reinsert_frozen_segments(segment_starts, frozen_segment_starts, is_frozen):
    all_segment_starts = []
    shift = 0
    unfrozen_index = 0
    frozen_index = 0
    for i in range(len(is_frozen)):
        if is_frozen[i]:
            shift += 1
            if frozen_index < len(frozen_segment_starts) and frozen_segment_starts[frozen_index] == i:
                all_segment_starts.append(i)
                frozen_index += 1
        else:
            if unfrozen_index < len(segment_starts) and segment_starts[unfrozen_index] + shift == i:
                all_segment_starts.append(i)
                unfrozen_index += 1
    return all_segment_starts

def mean_shift_step(rd, hb, hr, is_frozen, frozen_segment_starts):
    unfrozen_rd = rd[not is_frozen]
    unfrozen_hr = hr[not is_frozen]
    shifted_rd = unfrozen_rd
    for i in range(3):
        gradient = mean_shift_gradient(shifted_rd, hb, unfrozen_hr)
        segment_starts, shifted_rd = mean_shift_partition(shifted_rd, gradient)
    all_segment_starts = reinsert_frozen_segments(segment_starts, frozen_segment_starts, is_frozen)
    return all_segment_starts


def definitely_not_cnvnator(path, chromosomes, bin_length):
    samfile = pysam.AlignmentFile(path, "rb")
    chromosome_lengths = get_chromosome_lengths(samfile, chromosomes)
    for chromosome in chromosomes:
        counts, starts, gc_percents = scan_chromosome(chromosome, samfile, bin_length, chromosome_lengths[chromosome])
        rd_corrected, rd_global = gc_correction(counts, gc_percents)
        print(np.unique(rd_corrected))
        #mean_shift(counts, rd_corrected, rd_global)


def main():
    path ='../data/SRR_final_sorted.bam'
    chromosomes = ['chr21']
    # chromosomes = set([f'chr{i}' for i in range(1, 22)] + ['chrX'])
    bin_length = 300
    definitely_not_cnvnator(path, chromosomes, bin_length)

if __name__ == '__main__':
    main()
