import torch
from transformers import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Dict

from ..utils import pad_sequences


@dataclass
class NERCollator:
    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = field(default=None)
    pad_to_multiple_of: Optional[int] = field(default=None)
    padding: Union[bool, str, PaddingStrategy] = field(default="max_length")
    labels_key: str = field(default="labels")
    ignore_index: int = field(default=-100)

    def __call__(self, batch: Dict[str, Any]) -> Any:
        lengths = [len(sample["input_ids"]) for sample in batch]
        max_length = max(lengths)

        if self.max_length is not None:
            max_length = min(max_length, self.max_length)

        batch = self.tokenizer.pad(encoded_inputs=batch, 
                                   max_length=max_length, 
                                   padding=self.padding, 
                                   pad_to_multiple_of=self.pad_to_multiple_of)
            
        if "offset_mapping" in batch:
            batch["offset_mapping"] = pad_sequences(sequences=batch["offset_mapping"], 
                                                    max_length=max_length, 
                                                    padding_side=self.tokenizer.padding_side,
                                                    padding_value=(0, 0))
            
            
        if self.labels_key is not None and self.labels_key in batch:
            batch[self.labels_key] = pad_sequences(sequences=batch[self.labels_key], 
                                                   max_length=max_length, 
                                                   padding_side=self.tokenizer.padding_side,
                                                   padding_value=self.ignore_index)
            
        return {
                key: torch.tensor(value, dtype=torch.int32) if not isinstance(value[0], str) else value
                for key, value in batch.items()
            }