import os
import time
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

defaultconfig={
    'vocab': ['[PAD]', '[START]', '[END]', 'A', 'G', 'T', 'C', '[CpG]', '[CHG]', '[CHH]', '+', '-', '[MASK]', '[m6A]', '[R10]', '[4khz]', '[5khz]'],
    'num_workers':4,
    'batch_size':512,
    'num_bins': 15,
    'bin_size':500,
    'max_bin_length':512,
    'chunk_size':256,
    'overlap':16,
    'cpg':0,
    'chg':0,
    'chh':0,
    'm6A':0,
    'pore_type': 'r9.4.1',
    'frequency': '4khz',
    'mapq_thres': 0,
    'dorado_version': 0.71,
    'use_prune': 0,
}
methy_types = ['[CpG]', '[CHG]', '[CHH]', '[m6A]']
vocab = defaultconfig['vocab']
tokenizer = {vocab[i]: i for i in range(len(vocab))}
modelconfig = {
    "d_ff": 2048,
    "d_kv": 64,
    "d_model": 384,
    "max_position_embeddings": 4096,
    "dropout_rate": 0.1,
    "feed_forward_proj": "gated-gelu",
    "is_encoder_decoder": True,
    "num_heads": 8,
    "num_layers": 12,
    "pad_token_id": tokenizer['[PAD]'],
    "eos_token_id": tokenizer['[END]'],
    "decoder_start_token_id": tokenizer['[START]'],
    "vocab_size": len(vocab)
}

