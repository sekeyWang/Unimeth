import os
import argparse


basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}


def complement_seq(dnaseq):
    rdnaseq = dnaseq[::-1]
    comseq = ''
    try:
        comseq = ''.join([basepairs[x] for x in rdnaseq])
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


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            raise ValueError()

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        elif len(bases_list) == 2:
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


def get_c_motif2seq():
    motif2seq = dict()
    motif2seq["CGN"] = set(_convert_motif_seq("CGN"))
    motif2seq["CGN"].add("CGN")

    motif2seq["CHG"] = set(_convert_motif_seq("CHG"))
    motif2seq["CHG"].add("CHG")

    motif2seq["CHH"] = set(_convert_motif_seq("CHH"))
    motif2seq["CHH"].add("CHH")
    return motif2seq


def get_motifseq(chrom, pos, strand, contigs):
    if strand == "+":
        seq = contigs[chrom][pos:(pos+3)]
    else:
        seq = complement_seq(contigs[chrom][(pos-2):(pos+1)])
    return seq


def _split_callmods_file(cmfile, ref):
    fname, fext = os.path.splitext(cmfile)

    motif2seq = get_c_motif2seq()
    motifs = list(motif2seq.keys())
    seq2motif = dict()
    for motif in motifs:
        seqs = list(motif2seq[motif])
        for seq in seqs:
            seq2motif[seq] = motif

    motif2idx = {motifs[i]: i for i in range(len(motifs))}
    wfobjs = []
    for motif in motifs:
        if motif.startswith("CG"):
            motifstr = "CG"
        else:
            motifstr = motif
        if fname.endswith(".freq"):
            wfobjs.append(open(fname.rstrip(".freq") + "." + motifstr + ".freq" + fext, "w"))
        elif fname.endswith(".frequency"):
            wfobjs.append(open(fname.rstrip(".frequency") + "." + motifstr + ".frequency" + fext, "w"))
        else:
            wfobjs.append(open(fname + "." + motifstr + fext, "w"))

    count, count_fail = 0, 0
    
    # call_mods.tsv
    # chrom, pos, strand, _pos_in_strand, readid, readloc, unmod, mod, label, kmer/.
    if ref is None:
        raise ValueError("--ref must be providedd!")
    contigs = DNAReference(ref).getcontigs()
    with open(cmfile, "r") as rf:
        for line in rf:
            count += 1
            words = line.strip().split("\t")
            chrom = words[0]
            pos = int(words[1])
            strand = words[2]
            if pos < 0:
                print("invalid pos: {}, line: {}".format(pos, line.strip()))
                continue
            if strand == "+":
                seq = contigs[chrom][pos:(pos+3)]
            else:
                seq = complement_seq(contigs[chrom][(pos-2):(pos+1)])
            if len(seq) < 3:
                print("chrom: {}, pos: {}, seq: {}".format(chrom, pos, seq))
                continue
            # if seq[0] != "C":
            #     seq = complement_seq(contigs[chrom][(pos-2):(pos+1)])
            try:
                wfobjs[motif2idx[seq2motif[seq]]].write(line)
            except KeyError:
                count_fail += 1
                print("seq: {}, line: {}".format(seq, line.strip()))

    for wf in wfobjs:
        wf.flush()
        wf.close()
    print("total lines: {}, failed lines: {}".format(count, count_fail))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--callmods_file", type=str, required=True,
                        help="callmods file of deepsignals")
    parser.add_argument("--ref", type=str, required=True,
                        help="reference genome file")

    args = parser.parse_args()
    _split_callmods_file(args.callmods_file, args.ref)


if __name__ == '__main__':
    main()
