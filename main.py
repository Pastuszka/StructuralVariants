import numpy as np
import pysam
from scipy.stats import norm
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from Bio import SeqIO

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
        iter = samfile.fetch(name, start=start, stop=start + bin_length)
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
    n = len(segments)
    averages = np.empty(n)
    sds = np.empty(n)
    ns = np.empty(n)
    segment_edges = segments
    segment_edges.append(n)
    genome_length = len(rd_corrected)
    for i in range(n):
        averages[i] = np.mean(rd_corrected[segment_edges[i]:segment_edges[i+1]])
        sds[i] = np.std(rd_corrected[segment_edges[i]:segment_edges[i+1]])
        ns[i] = segment_edges[i+1] - segment_edges[i]
    for i in range(n):
        t1 = ((average_rd - averages[i]) / sds[i]) * np.sqrt(ns[i])
        p1 = 2*(t.cdf(-abs(t1), ns[i] - 1))
        segment_length = segment_edges[i+1] - segment_edges[i]
        p_corr1 = (p1 * 0.99 * genome_length) / segment_length
        cond1 = p_corr1 < 0.05
        if i > 0:
            mean_diff = averages[i-1] - averages[i]
            s1 = sds[i-1]
            s2 = sds[i]
            n1 = segment_edges[i] - segment_edges[i-1]
            n2 = segment_length
            t2 = mean_diff/np.sqrt(s1**2/n1 + s2**2/n2)
            #t2 = (mean_diff / sds[i]) * np.sqrt(2)
            p2 = 2 * (t.cdf(-abs(t2), 1))
            p2_corr = p2 * 0.99 * genome_length / (n1 + n2)
            cond2 = p2_corr < 0.01 or abs(mean_diff) >= 2 * hr[i]
        else:
            cond2 = True
        if i < n-1:
            mean_diff = averages[i+1] - averages[i]
            s1 = sds[i + 1]
            s2 = sds[i]
            n1 = segment_edges[i + 1] - segment_edges[i]
            n2 = segment_length
            t3 = mean_diff / np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
            #t3 = (mean_diff / sds[i]) * np.sqrt(2)
            p3 = 2 * (t.cdf(-abs(t3), 1))
            p3_corr = p3 * 0.99 * genome_length / (n1+n2)
            cond3 = p3_corr < 0.01 or abs(mean_diff) >= 2 * hr[i]
        else:
            cond3 = True
        if cond1 and cond2 and cond3:
            frozen_segment_starts.append(segments[i])
            is_frozen[segment_edges[i]:segment_edges[i+1]] = True
    return is_frozen, frozen_segment_starts


def mean_shift(counts, rd_corrected, rd_global, limit=128):
    _, h0 = norm.fit(rd_corrected)
    hr = np.array([np.sqrt(counts[i] / rd_global) * h0 if
          rd_corrected[i] > rd_global/4 else
          h0/2 for i in range(len(rd_corrected))])
    hb = 2
    is_frozen = np.full(len(counts), False)
    frozen_segment_starts = []
    while hb < limit:
        segments = mean_shift_step(rd_corrected, hb, hr, is_frozen, frozen_segment_starts)
        is_frozen, frozen_segment_starts = freeze_segments(segments, rd_corrected, hr)
        hb += 1
        print(hb)
    return segments


def mean_shift_gradient(rd, hb, hr, max_length=40000000):
    start = 0
    gradient = np.empty(len(rd))
    while start < len(rd):
        n = min(len(rd)-start, max_length)
        ind = np.indices([n, hb])
        ind += start
        look_right = ind[1] + ind[0] + 1
        look_right[look_right >= len(rd)] = ind[0][look_right >= len(rd)]
        look_left = ind[0] - ind[1] - 1
        look_left[look_left < 0] = ind[0][look_left < 0]
        i = np.hstack((ind[0], ind[0]))
        j = np.hstack((look_left, look_right))
        a = j - i
        b = np.exp(-(a ** 2) / (2 * (hb ** 2)))
        c = np.exp(-((rd[j] - rd[i]) ** 2) / (2 * (hr[i] ** 2)))
        gradient[start:(n+start)] = np.sum(a * b * c, axis=1)
        start = n
    return gradient


def mean_shift_partition(rd, gradient):
    start = 0
    segment_starts = [0]
    rd_new = np.empty_like(rd)
    for i in range(len(gradient)):
        if i == len(gradient) - 1:
            rd_new[start:(i + 1)] = np.mean(rd[start:(i + 1)])
        elif gradient[i] <= 0 < gradient[i + 1]:
            rd_new[start:(i + 1)] = np.mean(rd[start:(i + 1)])
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
    unfrozen_rd = rd[np.logical_not(is_frozen)]
    unfrozen_hr = hr[np.logical_not(is_frozen)]
    shifted_rd = unfrozen_rd
    for i in range(3):
        gradient = mean_shift_gradient(shifted_rd, hb, unfrozen_hr)
        segment_starts, shifted_rd = mean_shift_partition(shifted_rd, gradient)
    all_segment_starts = reinsert_frozen_segments(segment_starts, frozen_segment_starts, is_frozen)
    return all_segment_starts


def merge_signals(rd_corrected, segments):
    threshold = np.mean(rd_corrected) / 4
    new_segments = [0]
    segments.append(len(rd_corrected))
    for i in range(2, len(segments)):
        previous_average = np.mean(rd_corrected[new_segments[-1]:segments[i-1]])
        this_average = np.mean(rd_corrected[segments[i-1]:segments[i]])
        if np.abs(previous_average - this_average) > threshold:
            new_segments.append(segments[i-1])
    return new_segments


def call_variants(segments, rd_corrected):
    segments.append(len(rd_corrected))
    rd_global = np.mean(rd_corrected)
    calls = []
    is_deletion = []
    for i in range(1, len(segments)):
        start = segments[i-1]
        end = segments[i]
        rd_segment = np.mean(rd_corrected[start:end])
        s_segment = np.std(rd_corrected[start:end])
        n = end - start
        if s_segment == 0:
            calls.append((start, end-1))
            continue
        t_stat = ((rd_global-rd_segment)/s_segment)*np.sqrt(n)
        p = 2 * (t.cdf(-abs(t_stat), n - 1))
        p_corr = p * 0.99 * len(rd_corrected) / n
        if p_corr < 0.05:
            calls.append((start, end - 1))
            is_deletion = (rd_global-rd_segment) > 0
    return calls, is_deletion


def merge_calls(calls, rd_corrected, is_deletion):
    new_calls = []
    new_is_deletion = []
    genome_length = len(rd_corrected)
    for i in range(1, len(calls)):
        call_1 = rd_corrected[calls[i-1][0]:calls[i-1][1]]
        call_2 = rd_corrected[calls[i][0]:calls[i][1]]
        region = rd_corrected[calls[i-1][1]:calls[i][0]]
        mean_1 = np.mean(call_1)
        mean_2 = np.mean(call_2)
        mean_r = np.mean(region)

        std_1 = np.std(call_1)
        std_2 = np.std(call_2)
        std_r = np.std(region)
        n1 = len(call_1)
        n2 = len(call_2)
        nr = len(region)
        t1 = (mean_1-mean_r) / np.sqrt(std_1 ** 2 / n1 + std_r ** 2 / nr)
        t2 = (mean_2 - mean_r) / np.sqrt(std_2 ** 2 / n2 + std_r ** 2 / nr)
        p1 = 2 * (t.cdf(-abs(t1), 1))
        p2 = 2 * (t.cdf(-abs(t2), 1))
        p1_corr = p1 * 0.01 * genome_length / (n1 + nr)
        p2_corr = p2 * 0.01 * genome_length / (n2 + nr)
        if p1_corr > 0.01 and p2_corr > 0.01:
            calls[i] = (calls[i-1][0], calls[i][1])
        else:
            new_calls.append(calls[i-1])
            new_is_deletion.append(is_deletion[i-1])

    new_calls.append(calls[-1])
    is_deletion.append(is_deletion[-1])
    return new_calls, new_is_deletion


def jank_shift(rd_corrected):
    data = np.vstack((np.arange(len(rd_corrected)), rd_corrected)).T
    meanshift = MeanShift(n_jobs=-1, max_iter=2, min_bin_freq=50)
    return meanshift.fit_predict(data)


class VariantCall:
    def __init__(self, chrom, pos, end, ref, is_deletion):
        self.chrom = chrom
        self.pos = pos
        self.end = end
        self.ref = ref
        if is_deletion:
            self.type = '<DEL>'
        else:
            self.type = '<INS>'

    def __str__(self):
        line = self.chrom + '\t' + str(self.pos) + '\t' + '.\t' + str(self.ref) + '\t' + self.type + \
               '\t.\tLowQual\tIMPRECISE;SVMETHOD=HEMORRHAGEv0.0.0.0.0.1;SVLEN=' + str(self.end-self.pos) + ';'
        return line


def write_vcf(filename, calls):
    with open(filename, 'w') as f:
        f.write('##fileformat=VCFv4.2\n')
        f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
        for call in calls:
            print(call, file=f)


def create_variant_calls(reference, calls, bin_length, chromosome, trim_length, is_deletion):
    variant_calls = []
    sequence = reference[chromosome]
    for call in calls:
        start = call[0] * bin_length + trim_length
        end = call[1] * bin_length + trim_length
        ref = sequence[start]
        variant_call = VariantCall(chromosome, start, end, ref, is_deletion)
        variant_calls.append(variant_call)
    return variant_calls


def snip_beginning(counts):
    i = 0
    for read in counts:
        if read == 0:
            i += 1
        else:
            break
    return i, counts[i:len(counts)]


def definitely_not_cnvnator(path, chromosomes, bin_length):
    samfile = pysam.AlignmentFile(path, "rb")
    chromosome_lengths = get_chromosome_lengths(samfile, chromosomes)
    reference = SeqIO.index("../data/reference/GRCh38_full_analysis_set_plus_decoy_hla.fa", "fasta")
    variant_calls = []
    for chromosome in chromosomes:
        counts, starts, gc_percents = scan_chromosome(chromosome, samfile, bin_length, chromosome_lengths[chromosome])
        trim_length, counts_trimmed = snip_beginning(counts)
        rd_corrected, rd_global = gc_correction(counts, gc_percents)
        print(np.unique(rd_corrected))
        segments = mean_shift(counts, rd_corrected, rd_global)
        merged_segments = merge_signals(rd_corrected, segments)
        #segments = jank_shift(rd_corrected)
        calls, is_deletion = call_variants(merged_segments, rd_corrected)
        merged_calls, is_deletion_merged = merge_calls(calls, rd_corrected, is_deletion)
        variant_calls.extend(create_variant_calls(reference, merged_calls, bin_length, chromosome, trim_length, is_deletion_merged))
        plt.plot(rd_corrected)
        for call in calls:
            plt.axvspan(call[0], call[1], facecolor='red', alpha=.2)
        #plt.vlines(merged_segments, ymin=np.min(rd_corrected), ymax=np.max(rd_corrected), colors="b", linestyles='dotted')
        plt.savefig('test.png')
    write_vcf('hemorrhage.vcf', variant_calls)


def main():
    path = '../data/SRR_final_sorted.bam'
    chromosomes = ['chr21']
    # chromosomes = set([f'chr{i}' for i in range(1, 22)] + ['chrX'])
    bin_length = 100
    definitely_not_cnvnator(path, chromosomes, bin_length)


if __name__ == '__main__':
    main()
