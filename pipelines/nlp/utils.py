import torch
import numpy as np
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput
from typing import Callable, List, Any, Union, Optional, Callable

from .enums import PaddingSide


def pad_sequence(sequence: List[int], 
                 max_length: int, 
                 padding_value: Any = -1, 
                 padding_side: Union[PaddingSide, str] = "right"
                 ) -> List[int]:
    
    padding_side = PaddingSide(padding_side)

    sequence_length = len(sequence)
    length_diff = max_length - sequence_length

    padding_value = [padding_value]
    padding_values = padding_value * length_diff

    if padding_side == PaddingSide.LEFT:
        return padding_values + sequence
    
    return sequence + padding_values


def pad_sequences(sequences: List[List[int]], **kwargs) -> List[List[int]]:
    return [pad_sequence(sequence=sequence, **kwargs) for sequence in sequences]


def _get_embeddings(outputs: ModelOutput):
    return outputs[0]

def sliding_window(model: PreTrainedModel, 
                   input_ids: Union[torch.Tensor, np.ndarray], 
                   attention_mask: Optional[Union[torch.Tensor, np.ndarray]] = None, 
                   window_size: int = 512,
                   edge_length: int = 64,
                   inner_length: Optional[int] = None,
                   get_embeddings: Callable[[Any], Any] = _get_embeddings,
                   **kwargs,
                   ) -> torch.Tensor:

    """
    References:
        https://www.kaggle.com/code/aerdem4/xgb-lgb-feedback-prize-cv-0-7322/notebook#Network
    """

    if inner_length is None:
        inner_length = window_size - edge_length * 2

    batch_size, length = input_ids.shape

    if length <= window_size:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        outputs = get_embeddings(outputs)
    else:
        segments = (length - window_size) // inner_length
        if (length-window_size) % inner_length > edge_length:
            segments += 1
        elif segments == 0:
            segments += 1

        outputs = model(input_ids=input_ids[:,:window_size], attention_mask=attention_mask[:,:window_size], **kwargs)
        outputs = get_embeddings(outputs)
        
        for i in range(1, segments + 1):
            start = window_size - edge_length + (i - 1) * inner_length
            end = window_size - edge_length + (i - 1) * inner_length + window_size
            end = min(end, length)
            outputs_next = input_ids[:,start:end]
            attention_mask_next = attention_mask[:,start:end]

            outputs_next = model(input_ids=outputs_next, attention_mask=attention_mask_next, **kwargs)
            outputs_next = get_embeddings(outputs_next)        

            if i == segments:
                outputs_next = outputs_next[:, edge_length:]
            else:
                outputs_next = outputs_next[:, edge_length:edge_length + inner_length]
            
            outputs = torch.cat([outputs, outputs_next], dim=1)

    return outputs
    