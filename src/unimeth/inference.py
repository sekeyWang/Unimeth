from torch.utils.data import DataLoader
from model.data import collate_fn_inference, RawDataset
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from model.model import Basecaller
from config import defaultconfig, vocab
from util import local_print
from pathlib import Path

def get_result(decoder_input_ids, logits, patch_pos): 
    num_data = logits.shape[0]
    preds, methy = [], []
    for i in range(num_data):
        logits_patch = logits[i]
        pos = patch_pos[i]
        site_logits = logits_patch[pos]
        preds_patch = torch.exp(site_logits[:, 10]) / torch.sum(torch.exp(site_logits[:, [10, 11]]), axis = 1)
        methy_patch = decoder_input_ids[i][pos]
        preds.append(preds_patch)
        methy.append(methy_patch)
    return torch.hstack(preds), torch.hstack(methy)


def inference(args):
    dataLoaderConfiguration = DataLoaderConfiguration(
        dispatch_batches=False
    )
    accelerator = Accelerator(
        dataloader_config=dataLoaderConfiguration
    )
    dataset = RawDataset(pod5_dir=args.pod5_dir,bam_dir=args.bam_dir,args=args)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn_inference,
        batch_size=args.batch_size,  
        num_workers=args.num_workers,  
        pin_memory=True,  
        prefetch_factor=2  
    )
    dataloader = accelerator.prepare(dataloader)
    model = Basecaller(mode='inference')
    model.load_state_dict(torch.load(args.model_dir), strict=True)
    model.eval()
    model = accelerator.prepare(model)
    
    file_path = Path(args.out_dir)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), torch.cuda.amp.autocast(), open(args.out_dir, 'w') as fw:
        for num_batch, batch in enumerate(dataloader):
            if args.limit is not None and num_batch > args.limit:
                break
            logits = model(
                signals = batch['signals'],
                encoder_mask = batch['encoder_mask'],
                decoder_input_ids = batch['decoder_input_ids'],
                signal_pos = batch['signal_pos'],
            )
            logits = accelerator.pad_across_processes(logits, dim=1, pad_index=0)
            decoder_input_ids = accelerator.pad_across_processes(batch['decoder_input_ids'], dim=1, pad_index=0)
            logits = accelerator.gather(logits).cpu()
            decoder_input_ids = accelerator.gather(decoder_input_ids).cpu()
            patch_pos = accelerator.gather_for_metrics(batch['patch_pos'])
            read_id = accelerator.gather_for_metrics(batch['read_id'])
            chr = accelerator.gather_for_metrics(batch['chr'])
            strand = accelerator.gather_for_metrics(batch['strand'])
            ref_pos = accelerator.gather_for_metrics(batch['ref_pos'])
            read_pos = accelerator.gather_for_metrics(batch['read_pos'])
            labels = accelerator.gather_for_metrics(batch['labels'])

            preds, methy = get_result(decoder_input_ids, logits, patch_pos)
            idx = 0
            for i in range(len(read_id)):
                patch_pos_i = patch_pos[i]
                for j in range(len(patch_pos_i)):
                    prob1 = preds[idx].item()
                    prob0 = 1 - prob1
                    probbool = int(prob1 > 0.5)
                    # line = [chr[i], str(ref_pos[i][j]), strand[i], str(labels[i][j]), read_id[i], str(read_pos[i][j]), 
                    #         vocab[methy[idx].item()], str(prob0), str(prob1), str(probbool), '.']
                    # disable methy[idx].item() temporarily to make it compatible with post-processing scripts
                    line = [chr[i], str(ref_pos[i][j]), strand[i], str(labels[i][j]), read_id[i], str(read_pos[i][j]), 
                            vocab[methy[idx].item()], str(prob0), str(prob1), str(probbool), '.']
                    try:
                        line = '\t'.join(line)
                    except:
                        print(line)
                    fw.write(line + '\n')
                    idx += 1
            assert idx == len(preds)

def main(args):
    for key, value in defaultconfig.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    args.mode='inference'
    local_print(args)
    inference(args)