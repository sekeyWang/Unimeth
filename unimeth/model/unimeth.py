import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import shift_tokens_right
from unimeth.model.nn_modules import SignalProcessor, PositionalEncoding
from unimeth.config import default_modelconfig


class UniMeth(nn.Module):
    """
    UniMeth: Unified DNA Methylation Detection Model
    
    Args:
        mode: 'pretrain', 'finetune', 'inference', or 'distillation'
        plant: Use plant-specific unbalanced loss
        config: Model configuration dict (default: default_modelconfig)
    """
    def __init__(self, mode, plant=False, config=default_modelconfig):
        super().__init__()
        self.config = config
        
        # Signal processor
        self.processor = SignalProcessor(config)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config['d_model']
        )
        
        # BART encoder-decoder
        bart_config = BartConfig(**config)
        self.encoder_decoder = BartForConditionalGeneration(bart_config)
        
        self.entropy_loss = CrossEntropyLoss(reduction='none')
        self.mode = mode
        self.plant = plant
        # Methylation token IDs from unimeth.config (auto-derived from vocab)
        self.methy_types = config['methylation_tokens']
    
    def get_model_info(self):
        """Return model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 ** 2)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'hidden_size': self.config['d_model'],
            'num_layers': self.config['num_layers'],
            'cnn_downsample': self.processor.total_stride,
        }
    
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
            decoder_mask = torch.ones(size=(batch_size, seq_length), device=device)
            
        outputs = self.encoder_decoder(
            inputs_embeds=x, 
            attention_mask=encoder_mask, 
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_mask,
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
        methy_label_tokens = self.config['methylation_labels']
        for methy_token in self.methy_types:
            mask1 = decoder_input_ids == methy_token
            for methy_label in methy_label_tokens:
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
