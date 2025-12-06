'''
    Extract features from given (bam_read, pod5_read) pair
'''

import numpy as np

def get_chr_list(chr_str):
    if '|' not in chr_str:
        print('error, default as using all chr')
        return 'exclude', set()
    parts = chr_str.split('|')
    if parts[0] != '' and parts[1] != '':
        print('error, default as using all chr')
        return 'exclude', set()
    elif parts[0] == '' and parts[1] == '':
        chr_mode, chr_list = 'exclude', set()
    elif parts[0] != '':
        chr_mode, chr_list = 'exclude', set(parts[0].split(','))
    else:
        chr_mode, chr_list = 'include', set(parts[1].split(','))
    return chr_mode, chr_list

def get_ref_pos(seq, pred_pos, aligned_pairs, is_reverse):
    '''
        Given read sequence, aligned_pairs and if it is reverse
        Output reference position, len(seq) = len(ref_pos)
    '''
    ref_pos = []
    if len(pred_pos) == 0 or len(aligned_pairs) == 0:
        return [-1] * len(pred_pos)
    idx = 0
    for ps, pr in aligned_pairs:
        if ps is None:
            continue
        ps = len(seq) - ps - 1 if is_reverse else ps
        if ps == pred_pos[idx]:
            ref_pos.append(pr if pr is not None else -1)
            idx += 1
        elif ps > pred_pos[idx]:
            ref_pos.append(-1)
            idx += 1
        if idx == len(pred_pos):
            break
    assert len(ref_pos) == len(pred_pos), f'{len(ref_pos)} and {len(pred_pos)}'
    return ref_pos


basepairs = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}

def alphabet(letter):
    if letter in basepairs.keys():
        return basepairs[letter]
    return "N"

def complement_seq(base_seq):
    rbase_seq = base_seq[::-1]
    comseq = ""
    try:
        comseq = "".join([alphabet(x) for x in rbase_seq])
    except Exception:
        print("something wrong in the dna/rna sequence.")
    return comseq

def align_to_ref(bam_read, seq, signal_event, aligned_pairs, mod_dict):
    try:
        ref_seq = bam_read.get_reference_sequence().upper()
    except:
        return None
    if bam_read.is_reverse:
        ref_seq = complement_seq(ref_seq)
    ref_start = bam_read.reference_start
    first_index = next((i for i in range(len(aligned_pairs)) if aligned_pairs[i][1] is not None), len(aligned_pairs))
    last_index = next((i for i in range(len(aligned_pairs)-1, 0, -1) if aligned_pairs[i][1] is not None), len(aligned_pairs))
    new_ref_seq, new_ref_signal, new_ref_pos = [], [], []
    new_mod_dict = {}
    for i, (ps, pr) in enumerate(aligned_pairs):
        if ps is not None and bam_read.is_reverse:
            ps = len(seq) - ps - 1
        signal = [] if ps is None or signal_event is None else signal_event[ps]
        if i < first_index or i > last_index:
            if ps is not None:
                if ps in mod_dict:
                    new_mod_dict[len(new_ref_seq)] = mod_dict[ps]
                new_ref_seq.append(seq[ps])
                new_ref_signal.append(signal)
                new_ref_pos.append(pr)
        else:
            if pr is not None:
                if bam_read.is_reverse:
                    ref_base = ref_seq[len(ref_seq) - (pr - ref_start) - 1] 
                else:
                    ref_base = ref_seq[pr - ref_start]
                if ps in mod_dict:
                    new_mod_dict[len(new_ref_seq)] = mod_dict[ps]
                new_ref_seq.append(ref_base)
                new_ref_signal.append(signal)
                new_ref_pos.append(pr)
            elif len(new_ref_signal) > 0:
                new_ref_signal[-1] = np.append(new_ref_signal[-1], signal)
    seq = ''.join(new_ref_seq)
    signal_event = new_ref_signal
    if bam_read.is_reverse:
        aligned_pairs = [(len(new_ref_seq) - 1 - i, new_ref_pos[i]) for i in range(len(new_ref_seq))]
    else:
        aligned_pairs = [(i, new_ref_pos[i]) for i in range(len(new_ref_seq))]
    return seq, signal_event, aligned_pairs, new_mod_dict

class Pred_pos_extractor:
    def __init__(self, args):
        self.detect_CpG = args.cpg
        self.detect_CHG = args.chg
        self.detect_CHH = args.chh
        self.detect_m6A = args.m6A

    def get_methy_pos(self, seq):
        pred_pos = []
        for i in range(len(seq)):
            if seq[i] == 'C':
                if i + 1 < len(seq) and seq[i+1] == 'G' and self.detect_CpG:
                    pred_pos.append(i)
                elif i + 2 < len(seq) and seq[i+2] == 'G' and self.detect_CHG:
                    pred_pos.append(i)
                elif i + 2 < len(seq) and self.detect_CHH:
                    pred_pos.append(i)
            elif seq[i] == 'A':
                if self.detect_m6A:
                    pred_pos.append(i)
        return pred_pos

class Extractor_raw:
    def __init__(self, args):
        self.mapq_thres = args.mapq_thres
        self.pred_pos_extractor = Pred_pos_extractor(args)
        if args.m6A == 1:
            self.detect_mod = ('A',0,'a')
        elif args.cpg or args.chg or args.chh:
            self.detect_mod = ('C',0,'m')
        else:
            self.detect_mod = None
        self.align_ref = (args.pore_type == 'R9.4.1')
        self.chr_mode, self.chr_list = get_chr_list(args.chr)
    
    #@jit(nopython=True)
    def get_signal(self, signal, move):
        stride, start, move_table=move
        move_index=np.where(move_table)[0]
        event=[]
        for i in range(len(move_index) - 1):
            prev=move_index[i]*stride+start
            sig_end=move_index[i+1]*stride+start
            event.append(signal[prev:sig_end])
        prev=move_index[len(move_index) - 1]*stride+start
        event.append(signal[prev:])
        return event

    def get_move(self, bam_read):
        read_dict = bam_read.to_dict()
        tags={x.split(':')[0]:x for x in read_dict.pop('tags')}
        start=int(tags['ts'].split(':')[-1])
        if bam_read.has_tag('sp'):
            start += bam_read.get_tag('sp')
        mv=tags['mv'].split(',')
        stride=int(mv[1])
        move_table=np.fromiter(mv[2:], dtype=np.int8)
        move=(stride, start, move_table)
        return move

    def get_feature(self, bam_read, pod5_read):
        mapq = bam_read.mapping_quality
        if mapq < self.mapq_thres:
            print(f'MapQ < {self.mapq_thres}')
            return None
        chr = bam_read.reference_name
        if chr is None:
            return None
        elif self.chr_mode == 'exclude' and chr in self.chr_list:
            return None
        elif self.chr_mode == 'include' and chr not in self.chr_list:
            return None

        seq = bam_read.get_forward_sequence().upper()

        signal = pod5_read.signal
        shift_dacs_to_pa=pod5_read.calibration.offset
        scale_dacs_to_pa=pod5_read.calibration.scale
        shift_pa_to_norm = bam_read.get_tag("sm")
        scale_pa_to_norm = bam_read.get_tag("sd")
        
        move = self.get_move(bam_read)
        signal_event = self.get_signal(signal, move)
        aligned_pairs = bam_read.get_aligned_pairs()
        mod = bam_read.modified_bases_forward
        mod_type = mod[self.detect_mod] if self.detect_mod in mod else []
        mod_dict = {}
        for pos, label in mod_type:
            mod_dict[pos] = label
        
        if bam_read.is_reverse:
            aligned_pairs = aligned_pairs[::-1]
        if self.align_ref:
            seq, signal_event, aligned_pairs, mod_dict = align_to_ref(bam_read, seq, signal_event, aligned_pairs, mod_dict)

        pred_pos = self.pred_pos_extractor.get_methy_pos(seq)
        ref_pos = get_ref_pos(seq, pred_pos, aligned_pairs, bam_read.is_reverse)

        bis_label = []
        for pos in pred_pos:
            if pos in mod_dict:
                bis_label.append(int(round(mod_dict[pos] / 255 * 100)))
            else:
                bis_label.append(-1)

        output = {
            'read_id': bam_read.query_name,
            'chr': chr,
            'reference_start': bam_read.reference_start,
            'bases': list(seq),
            'signal_event': signal_event,
            'pred_pos': pred_pos,
            'bis_label': bis_label,
            'mapQ': mapq,
            'shift_dacs_to_pa': shift_dacs_to_pa,
            'scale_dacs_to_pa': scale_dacs_to_pa,
            'shift_pa_to_norm': shift_pa_to_norm,
            'scale_pa_to_norm': scale_pa_to_norm,
            'ref_pos': ref_pos,
            'strand': '-' if bam_read.is_reverse else '+',
        }
        return output


