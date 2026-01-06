#! /usr/bin/env python
"""
calculate modification frequency at genome level
"""

import argparse
import os
import sys

import math
import gzip

import multiprocessing as mp
import multiprocessing.queues
import uuid
import time

time_wait = 0.1  # seconds


key_sep = "||"


class ModRecord:
    def __init__(self, fields):
        self._chromosome = fields[0]
        self._pos = int(fields[1])
        self._site_key = key_sep.join([self._chromosome, str(self._pos)])

        self._strand = fields[2]
        self._pos_in_strand = int(fields[3]) if fields[3] != '.' else -1
        self._readname = fields[4]
        self._read_loc = fields[5]
        self._prob_0 = float(fields[7])
        self._prob_1 = float(fields[8])
        self._called_label = int(fields[9])
        self._kmer = fields[10]

    def is_record_callable(self, prob_threshold):
        if abs(self._prob_0 - self._prob_1) < prob_threshold:
            return False
        return True
    
    def get_called_label(self, methyl_threshold=0.5):
        if self._prob_1 >= methyl_threshold:
            return 1
        else:
            return 0


def split_key(key):
    words = key.split(key_sep)
    return words[0], int(words[1])


class SiteStats:
    def __init__(self, strand, pos_in_strand, kmer):

        self._strand = strand
        self._pos_in_strand = pos_in_strand
        self._kmer = kmer

        self._prob_0 = 0.0
        self._prob_1 = 0.0
        self._met = 0
        self._unmet = 0
        self._coverage = 0
        # self._rmet = -1.0



# --- for parallel processing ---
# https://thispointer.com/python-three-ways-to-check-if-a-file-is-empty/
def is_file_empty(file_name):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.isfile(file_name) and os.path.getsize(file_name) == 0


class SharedCounter(object):
    """ A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):
        self.count = mp.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


# https://github.com/vterron/lemon/commit/9ca6b4b1212228dbd4f69b88aaf88b12952d7d6f
class MyQueue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, *args, **kwargs):
        super(MyQueue, self).__init__(*args, ctx=mp.get_context(), **kwargs)
        self._size = SharedCounter(0)

    def put(self, *args, **kwargs):
        super(MyQueue, self).put(*args, **kwargs)
        self._size.increment(1)

    def get(self, *args, **kwargs):
        self._size.increment(-1)
        return super(MyQueue, self).get(*args, **kwargs)

    def qsize(self) -> int:
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self._size.value

    def empty(self) -> bool:
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return self.qsize() == 0




alphabet = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def _alphabet(letter):
    if letter in alphabet.keys():
        return alphabet[letter]
    return 'N'


def complement_seq(dnaseq):
    rdnaseq = dnaseq[::-1]
    comseq = ''
    try:
        comseq = ''.join([_alphabet(x) for x in rdnaseq])
    except Exception:
        print('something wrong in the dna sequence.')
    return comseq


class DNAReference:
    def __init__(self, reffile):
        self._contignames = []
        self._contigs = {}  # contigname 2 contigseq
        with open(reffile, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        self._contigs[contigname] = contigseq
                        self._contignames.append(contigname)
                    contigname = line.strip()[1:].split(' ')[0]
                    contigseq = ''
                else:
                    # turn to upper case
                    contigseq += line.strip().upper()
            self._contigs[contigname] = contigseq
            self._contignames.append(contigname)

    def getcontigs(self):
        return self._contigs

    def getcontignames(self):
        return self._contignames


def convert_motif_seq(ori_seq):
    alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                 'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                 'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                 'N': ['A', 'C', 'G', 'T']}
    outbases = []
    for bbase in ori_seq:
        outbases.append(alphabets[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        if len(bases_list) == 2:
            pseqs = []
            for fbase in bases_list[0]:
                for sbase in bases_list[1]:
                    pseqs.append(fbase + sbase)
            return pseqs
        else:
            pseqs = recursive_permute(bases_list[1:])
            pseq_list = [bases_list[0], pseqs]
            return recursive_permute(pseq_list)
    return recursive_permute(outbases)


def calculate_mods_frequency(mods_files, prob_cf, methyl_threshold=0.5, 
                             contigs=None, motifset=None, mloc=0, motiflen=None, 
                             contig_name=None):
    sitekeys = set()
    sitekey2stats = dict()

    if type(mods_files) is str:
        mods_files = [mods_files, ]

    count, used = 0, 0
    for mods_file in mods_files:
        if mods_file.endswith('.gz'):
            rf = gzip.open(mods_file, 'rt')
        else:
            rf = open(mods_file, 'r')
        with rf:
            for line in rf:
                words = line.strip().split("\t")
                mod_record = ModRecord(words)
                if contig_name is not None and mod_record._chromosome != contig_name:
                    continue
                if mod_record._pos < 0:
                    continue
                # check if the site is in targeted motifs
                if contigs is not None and motifset is not None:
                    p_s = mod_record._pos - mloc
                    p_e = mod_record._pos + motiflen - mloc
                    if mod_record._strand == "+":
                        if contigs[mod_record._chromosome][p_s:p_e] not in motifset:
                            # print("skip line cause <not targeted motif in genome>: {}".format(line.strip()))
                            continue
                    else:
                        p_s = mod_record._pos - (motiflen - mloc) + 1
                        p_e = mod_record._pos + mloc + 1
                        if complement_seq(contigs[mod_record._chromosome][p_s:p_e]) not in motifset:
                            # print("skip line cause <not targeted motif in genome>: {}".format(line.strip()))
                            continue
                # -------------------------------------
                
                if mod_record.is_record_callable(prob_cf):
                    if mod_record._site_key not in sitekeys:
                        sitekeys.add(mod_record._site_key)
                        sitekey2stats[mod_record._site_key] = SiteStats(mod_record._strand,
                                                                        mod_record._pos_in_strand,
                                                                        mod_record._kmer)
                    sitekey2stats[mod_record._site_key]._prob_0 += mod_record._prob_0
                    sitekey2stats[mod_record._site_key]._prob_1 += mod_record._prob_1
                    sitekey2stats[mod_record._site_key]._coverage += 1
                    if mod_record.get_called_label(methyl_threshold) == 1:
                        sitekey2stats[mod_record._site_key]._met += 1
                    else:
                        sitekey2stats[mod_record._site_key]._unmet += 1
                    used += 1
                count += 1
    if contig_name is None:
        print("{:.2f}% ({} of {}) calls used..".format(used/float(count) * 100, used, count))
    else:
        print("{:.2f}% ({} of {}) calls used for {}..".format(used / float(count) * 100, used, count, contig_name))
    return sitekey2stats


def write_sitekey2stats(sitekey2stats, result_file, is_sort, is_bed, is_gzip, use_prob=False, use_floor=False):
    if is_sort:
        keys = sorted(list(sitekey2stats.keys()), key=lambda x: split_key(x))
    else:
        keys = list(sitekey2stats.keys())

    if is_gzip:
        if not result_file.endswith(".gz"):
            result_file += ".gz"
        wf = gzip.open(result_file, "wt")
    else:
        wf = open(result_file, 'w')

    for key in keys:
        chrom, pos = split_key(key)
        sitestats = sitekey2stats[key]
        assert(sitestats._coverage == (sitestats._met + sitestats._unmet))
        if sitestats._coverage > 0:
            if use_prob:
                rmet = float(sitestats._prob_1) / sitestats._coverage
            else:
                rmet = float(sitestats._met) / sitestats._coverage
            if is_bed:
                if use_floor:
                    wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(sitestats._coverage),
                                        sitestats._strand,
                                        str(pos), str(pos + 1), "0,0,0", str(sitestats._coverage),
                                        # str(int(round(rmet * 100 + 0.001, 0)))]) + "\n")
                                        str(int(math.floor(rmet * 100)))]) + "\n")
                else:
                    wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(sitestats._coverage),
                                        sitestats._strand,
                                        str(pos), str(pos + 1), "0,0,0", str(sitestats._coverage),
                                        str(int(round(rmet * 100 + 0.001, 0)))]) + "\n")
            else:
                wf.write("%s\t%d\t%s\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n" % (chrom, pos, sitestats._strand,
                                                                                    sitestats._pos_in_strand,
                                                                                    sitestats._prob_0,
                                                                                    sitestats._prob_1,
                                                                                    sitestats._met, sitestats._unmet,
                                                                                    sitestats._coverage, rmet,
                                                                                    sitestats._kmer))
        else:
            print("{} {} has no coverage..".format(chrom, pos))
    wf.flush()
    wf.close()


def _read_file_lines(cfile):
    with open(cfile, "r") as rf:
        return rf.read().splitlines()


def _get_contignams_from_genome_fasta(genomefa):
    contigs = []
    with open(genomefa, "r") as rf:
        for line in rf:
            if line.startswith(">"):
                contigname = line.strip()[1:].split(' ')[0]
                contigs.append(contigname)
    return contigs


def _is_file_a_genome_fasta(contigfile):
    with open(contigfile, "r") as rf:
        for line in rf:
            if line.startswith("#"):
                continue
            elif line.startswith(">"):
                return True
    return False


def _get_contigfile_name(wprefix, contig):
    return wprefix + "." + contig + ".txt"


def _split_file_by_contignames(mods_files, wprefix, contigs):
    contigs = set(contigs)
    wfs = {}
    for contig in contigs:
        wfs[contig] = open(_get_contigfile_name(wprefix, contig), "w")
    for input_file in mods_files:
        if input_file.endswith(".gz"):
            infile = gzip.open(input_file, 'rt')
        else:
            infile = open(input_file, 'r')
        for line in infile:
            chrom = line.strip().split("\t")[0]
            if chrom not in contigs:
                continue
            wfs[chrom].write(line)
        infile.close()
    for contig in contigs:
        wfs[contig].flush()
        wfs[contig].close()



def _call_and_write_modsfreq_process(wprefix, prob_cf, result_file, issort, isbed, isgzip,
                                     contigs_q, resfiles_q, 
                                     methyl_threshold, rcontigs, motifset, mloc, motiflen):
    print("process-{} -- starts".format(os.getpid()))
    while True:
        if contigs_q.empty():
            time.sleep(time_wait)
        contig_name = contigs_q.get()
        if contig_name == "kill":
            contigs_q.put("kill")
            break
        print("process-{} for contig-{} -- reading the input files..".format(os.getpid(), contig_name))
        input_file = _get_contigfile_name(wprefix, contig_name)
        if not os.path.isfile(input_file):
            print("process-{} for contig-{} -- the input file does not exist..".format(os.getpid(), contig_name))
            continue
        if is_file_empty(input_file):
            print("process-{} for contig-{} -- the input file is empty..".format(os.getpid(), contig_name))
        else:
            if rcontigs is None:
                sites_stats = calculate_mods_frequency(input_file, prob_cf, methyl_threshold, contig_name=contig_name)
            else:
                sites_stats = calculate_mods_frequency(input_file, prob_cf, methyl_threshold,
                                                       contigs=rcontigs, motifset=motifset,
                                                       mloc=mloc, motiflen=motiflen, 
                                                       contig_name=contig_name)
            print("process-{} for contig-{} -- writing the result..".format(os.getpid(), contig_name))
            fname, fext = os.path.splitext(result_file)
            c_result_file = fname + "." + contig_name + "." + str(uuid.uuid1()) + fext
            write_sitekey2stats(sites_stats, c_result_file, issort, isbed, isgzip)
            resfiles_q.put(c_result_file)
        os.remove(input_file)
    print("process-{} -- ends".format(os.getpid()))


def _concat_contig_results(contig_files, result_file, is_gzip=False):
    if is_gzip:
        if not result_file.endswith(".gz"):
            result_file += ".gz"
        wf = gzip.open(result_file, "wt")
    else:
        wf = open(result_file, 'w')
    for cfile in sorted(contig_files):
        with open(cfile, 'r') as rf:
            for line in rf:
                wf.write(line)
        os.remove(cfile)
    wf.close()


def call_mods_frequency_to_file(args):
    print("[main]call_freq starts..")
    start = time.time()

    input_paths = args.input_path
    result_file = args.result_file
    prob_cf = args.prob_cf
    file_uid = args.file_uid
    issort = args.sort
    isbed = args.bed
    is_gzip = args.gzip

    methyl_threshold = args.methyl_threshold
    use_prob = args.use_prob
    use_floor = args.use_floor

    if args.ref is not None:
        # read targeted sites from genome
        rcontigs = DNAReference(args.ref).getcontigs()
        # contignames = set(contigs.keys())

        motifset = set(convert_motif_seq(args.motif))
        motifbase = args.motif[args.mloc_in_motif]
        motiflen = len(args.motif)
    else:
        rcontigs = None
        motifset = None
        motiflen = None

    mods_files = []
    for ipath in input_paths:
        input_path = os.path.abspath(ipath)
        if os.path.isdir(input_path):
            for ifile in os.listdir(input_path):
                if file_uid is None:
                    mods_files.append('/'.join([input_path, ifile]))
                elif ifile.find(file_uid) != -1:
                    mods_files.append('/'.join([input_path, ifile]))
        elif os.path.isfile(input_path):
            mods_files.append(input_path)
        else:
            raise ValueError("--input_path is not a file or a directory!")
    print("get {} input file(s)..".format(len(mods_files)))

    contigs = None
    if args.contigs is not None:
        if os.path.isfile(args.contigs):
            if args.contigs.endswith(".fa") or args.contigs.endswith(".fasta") or args.contigs.endswith(".fna"):
                contigs = _get_contignams_from_genome_fasta(args.contigs)
            elif _is_file_a_genome_fasta(args.contigs):
                contigs = _get_contignams_from_genome_fasta(args.contigs)
            else:
                contigs = sorted(list(set(_read_file_lines(args.contigs))))
        else:
            contigs = sorted(list(set(args.contigs.strip().split(","))))

    if contigs is None:
        print("read the input files..")
        if rcontigs is None:
            sites_stats = calculate_mods_frequency(mods_files, prob_cf, methyl_threshold)
        else:
            sites_stats = calculate_mods_frequency(mods_files, prob_cf, methyl_threshold,
                                                   contigs=rcontigs, motifset=motifset,
                                                   mloc=args.mloc_in_motif, motiflen=motiflen)
        print("write the result..")
        write_sitekey2stats(sites_stats, result_file, issort, isbed, is_gzip, use_prob, use_floor)
    else:
        print("start processing {} contigs..".format(len(contigs)))
        wprefix = os.path.dirname(os.path.abspath(result_file)) + "/tmp." + str(uuid.uuid1())
        print("generate input files for each contig..")
        _split_file_by_contignames(mods_files, wprefix, contigs)
        print("read the input files of each contig..")
        contigs_q = MyQueue()
        for contig in contigs:
            contigs_q.put(contig)
        contigs_q.put("kill")
        resfiles_q = MyQueue()
        procs_contig = []
        for _ in range(args.nproc):
            p_contig = mp.Process(target=_call_and_write_modsfreq_process,
                                  args=(wprefix, prob_cf, result_file, issort, isbed, False,
                                        contigs_q, resfiles_q, 
                                        methyl_threshold, rcontigs, motifset, args.mloc_in_motif, motiflen))  # didn't gzip here, didn't use_prob and use_floor here
            p_contig.daemon = True
            p_contig.start()
            procs_contig.append(p_contig)
        resfiles_cs = []
        while True:
            running = any(p.is_alive() for p in procs_contig)
            while not resfiles_q.empty():
                resfiles_cs.append(resfiles_q.get())
            if not running:
                break
        for p in procs_contig:
            p.join()
        try:
            assert len(contigs) == len(resfiles_cs)
        except AssertionError:
            print("!!!Please check the result files -- seems not all inputed contigs have result!!!")
        print("combine results of {} contigs..".format(len(resfiles_cs)))
        _concat_contig_results(resfiles_cs, result_file, is_gzip)
    print("[main]call_freq costs %.1f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser(description='calculate frequency of interested sites at genome level')
    parser.add_argument('--input_path', '-i', action="append", type=str, required=True,
                        help='a result file from unimeth, or a directory contains a bunch of '
                             'result files.')
    parser.add_argument('--result_file', '-o', action="store", type=str, required=True,
                        help='the file path to save the result')
    parser.add_argument("--gzip", action="store_true", default=False, required=False,
                        help="if compressing the output using gzip")
    
    parser.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    parser.add_argument('--prob_cf', type=float, action="store", required=False, default=0.0,
                        help='this is to remove ambiguous calls. '
                             'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                             'means use all calls. range [0, 1], default 0.0.')
    parser.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                        help='a unique str which all input files has, this is for finding all input files and ignoring '
                             'the un-input-files in a input directory. if input_path is a file, ignore this arg.')
    
    parser.add_argument('--methyl_threshold', type=float, action="store", required=False, default=0.5)

    parser.add_argument("--use_prob", action="store_true", default=False, help="use the probability to call freq")
    parser.add_argument("--use_floor", action="store_true", default=False, help="use floor to round the rmet")

    parser.add_argument("--ref", type=str, required=False,
                        help="genome reference, .fasta or .fa, only for remove non-targeted motifs, not overlapped with --contigs")
    parser.add_argument("--motif", type=str, required=False, default="C",
                        help="targeted motif, default C")
    parser.add_argument('--mloc_in_motif', type=int, required=False,
                        default=0,
                        help='0-based location of the methylation base in the motif, default 0')
    
    parser.add_argument('--contigs', action="store", type=str, required=False, default=None,
                        help="a reference genome file (.fa/.fasta/.fna), used for extracting all "
                             "contig names for parallel; "
                             "or path of a file containing chromosome/contig names, one name each line; "
                             "or a string contains multiple chromosome names splited by comma."
                             "default None, which means all chromosomes will be processed at one time. "
                             "If not None, one chromosome will be processed by one subprocess.")
    parser.add_argument('--nproc', action="store", type=int, required=False, default=1,
                        help="number of subprocesses used when --contigs is set. i.e., number of contigs processed "
                             "in parallel. default 1")

    args = parser.parse_args()

    call_mods_frequency_to_file(args)


if __name__ == '__main__':
    sys.exit(main())
