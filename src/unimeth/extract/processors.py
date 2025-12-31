from extract.patcher import Patcher
from config import tokenizer
import numpy as np

class ReadProcessor:

    def __init__(self, args, negative_thres=0, positive_thres=100):
        self.mode = args.mode
        self.detect_CpG, self.detect_CHG, self.detect_CHH, self.detect_m6A = args.cpg, args.chg, args.chh, args.m6A
        self.pore_type, self.frequency, self.dorado_version = args.pore_type, args.frequency, args.dorado_version
        self.negative_thres, self.positive_thres = negative_thres, positive_thres
        self.patcher = Patcher(args)
        
    def weight_bis(self, num):
        return 10 * ((num - 50) / 50) ** 2

    def get_norm_params(self, feature):
        shift_dacs_to_pa, scale_dacs_to_pa = feature['shift_dacs_to_pa'], feature['scale_dacs_to_pa']
        shift_pa_to_norm, scale_pa_to_norm = feature['shift_pa_to_norm'], feature['scale_pa_to_norm']
        if self.frequency == '5khz':
            if self.dorado_version <= 0.71:
                shift = 1-shift_pa_to_norm
                scale = 1/scale_pa_to_norm
            else:
                shift = shift_pa_to_norm
                scale = scale_pa_to_norm
        else:
            shift = (shift_pa_to_norm / scale_dacs_to_pa) - shift_dacs_to_pa
            scale = scale_pa_to_norm / scale_dacs_to_pa
        return shift, scale
    
    def get_methy_type(self, seq, pos):
        if seq[pos] == 'C':
            if pos + 1 < len(seq) and seq[pos+1] == 'G':
                return '[CpG]' if self.detect_CpG else None
            elif pos + 2 < len(seq) and seq[pos+2] == 'G':
                return '[CHG]' if self.detect_CHG else None
            elif pos + 2 < len(seq):
                return '[CHH]' if self.detect_CHH else None
        elif seq[pos] == 'A':
            return '[m6A]' if self.detect_m6A else None

    def add_tokens(self, bases, signal_event, pred_pos, ref_pos=None, bis_label=None):
        meta_data = []
        for i in reversed(range(len(pred_pos))):
            bis_l = -1 if bis_label is None else bis_label[i]
            pos_r = None if ref_pos is None else ref_pos[i]
            pos_p = pred_pos[i]
            methy_type = self.get_methy_type(bases, pos_p)
            
            if methy_type is not None:
                bases.insert(pos_p+1, methy_type)
                signal_event.insert(pos_p+1, [])
                meta_data.append({
                    'ref_pos': pos_r, 
                    'labels': bis_l, 
                    'read_pos': pos_p,
                })     
        meta_data = meta_data[::-1]
        data_dict = {
            'bases': bases,
            'signal_event': signal_event,
        }
        return data_dict, meta_data

    def get_inference_datasets(self, feature):
        new_feature, meta_data = self.add_tokens(
            bases=feature['bases'], 
            signal_event=feature['signal_event'],
            pred_pos=feature['pred_pos'],
            ref_pos=feature['ref_pos'],
            bis_label=feature['bis_label']
        )
        
        shift, scale = self.get_norm_params(feature)
        patch_generator = self.patcher.get_patch_generator(
            mode=self.mode,
            bases=new_feature['bases'], 
            signal_event=new_feature['signal_event'],
            shift=shift,
            scale=scale,
            meta_data=meta_data,
        )
        for patch in patch_generator:
            if len(patch['patch_pos']) == 0:
                continue

            inference_data = {
                'read_id': feature['read_id'],
                'chr': feature['chr'],
                'strand': feature['strand'],
                'signals': patch['signals'],
                'signal_pos': patch['signal_pos'],
                'decoder_input_ids': patch['decoder_input_ids'],
                'patch_pos': patch['patch_pos'],
                'read_pos': patch['read_pos'],
                'ref_pos': patch['ref_pos'],
                'labels': patch['labels']
            }
            
            yield inference_data

    def get_finetune_datasets(self, feature):
        read_label = feature['bis_label']
        new_feature, meta_data = self.add_tokens(
            bases=feature['bases'], 
            signal_event=feature['signal_event'],
            pred_pos=feature['pred_pos'],
            bis_label=read_label
        )
        
        shift, scale = self.get_norm_params(feature)
        patch_generator = self.patcher.get_patch_generator(
            mode=self.mode,
            bases=new_feature['bases'], 
            signal_event=new_feature['signal_event'],
            shift=shift,
            scale=scale,
            meta_data=meta_data,
        )
        for patch in patch_generator:
            label_tokens, weight = [], []
            for x in patch['labels']:
                if x == -1:
                    label_tokens.append(tokenizer['[MASK]'])
                    weight.append(0)
                elif x <= self.negative_thres:
                    label_tokens.append(tokenizer['-'])
                    weight.append(self.weight_bis(x))
                elif x >= self.positive_thres:
                    label_tokens.append(tokenizer['+'])
                    weight.append(self.weight_bis(x))
                else:
                    label_tokens.append(tokenizer['[MASK]'])
                    weight.append(0)
            if (np.array(label_tokens) == tokenizer['[MASK]']).all():
                continue
            train_data = {
                'signals': patch['signals'],
                'signal_pos': patch['signal_pos'],
                'decoder_input_ids': patch['decoder_input_ids'],
                'patch_pos': patch['patch_pos'],
                'labels': label_tokens,
                'weight': weight,
            }
            yield train_data
        
    def get_pretrain_datasets(self, feature):
        shift, scale = self.get_norm_params(feature)
        patch_generator = self.patcher.get_patch_generator(
            mode=self.mode,
            bases=list(feature['bases']), 
            signal_event=feature['signal_event'],
            shift=shift,
            scale=scale,
        )
        for patch in patch_generator:
            train_data = {
                'signals': patch['signals'],
                'signal_pos': patch['signal_pos'],
                'decoder_input_ids': patch['decoder_input_ids']
            }
            yield train_data
    
    def get_datasets(self, feature):
        if self.mode == 'pretrain':
            yield from self.get_pretrain_datasets(feature)
        elif self.mode == 'finetune':
            yield from self.get_finetune_datasets(feature)
        elif self.mode == 'inference':
            yield from self.get_inference_datasets(feature)