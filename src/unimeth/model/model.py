import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right
from model.module import Processor, PositionalEncoding
from config import modelconfig

class Basecaller(nn.Module):
    def __init__(self, mode, plant=False):
        super().__init__()
        self.processor = Processor(hidden_size=modelconfig['d_model'])
        self.encoder_decoder = BartForConditionalGeneration(BartConfig(**modelconfig))
        self.pos_encoder = PositionalEncoding(d_model=modelconfig['d_model'])
        self.entropy_loss = CrossEntropyLoss(reduction='none')
        self.mode = mode
        self.plant=plant
        self.methy_types = [7, 8, 9, 13]
    
    def _forward(
            self, 
            signals: Tensor, 
            encoder_mask: Tensor, 
            signal_pos: Tensor, 
            decoder_input_ids: Tensor=None, 
            labels: Tensor=None, 
            ):
        x = self.processor(signals)
        x = self.pos_encoder(x, signal_pos)
        if self.mode == 'pretrain':
            decoder_mask = None
        else:
            decoder_input_ids = shift_tokens_right(decoder_input_ids, 0, 1)
            batch_size, seq_length = decoder_input_ids.shape
            device = decoder_input_ids.device
            decoder_mask = torch.ones(size=(batch_size, seq_length, seq_length), device=device)        
            
        outputs = self.encoder_decoder(
            inputs_embeds=x, 
            attention_mask=encoder_mask, 
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_mask,
            use_cache=False,
        )
        return outputs
    
    def forward(
            self, 
            signals: Tensor, 
            encoder_mask: Tensor, 
            signal_pos: Tensor, 
            decoder_input_ids: Tensor=None, 
            labels: Tensor=None, 
            weights: Tensor=None, 
        ):

        outputs = self._forward(
            signals = signals, 
            encoder_mask = encoder_mask,
            signal_pos = signal_pos,
            decoder_input_ids = decoder_input_ids,
            labels=labels,
        )
        logits, loss = outputs.logits, outputs.loss
        if self.mode == 'pretrain':
            return loss, None
        elif self.mode == 'finetune':
            if self.plant:
               return self.get_unbalance_loss(logits, labels, decoder_input_ids), logits
            else:
               return self.get_weighted_loss(logits, labels, weights), logits
                
        elif self.mode == 'inference':
            return logits
        

    def get_weighted_loss(self, logits, labels, weights):
        label_mask = (labels.view(-1) != -100)
        methy_loss = self.entropy_loss(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))
        weighted_loss = methy_loss * weights.view(-1)
        methy_loss = torch.mean(weighted_loss[label_mask])
        return methy_loss

    
    def get_unbalance_loss(self, logits, labels, decoder_input_ids):
        total_loss, total_items, new_loss = 0, 0, 0
        for methy_token in self.methy_types:
            mask1 = decoder_input_ids == methy_token
            for methy_label in [11, 10]:
                mask2 = (labels == methy_label)
                mask = mask1 & mask2
                if True not in mask:
                    continue
                subloss = self.entropy_loss(logits[mask], labels[mask]).mean()
                num_items = len(logits[mask])
                total_loss += subloss * num_items
                total_items += num_items
                new_loss += subloss
        return new_loss
