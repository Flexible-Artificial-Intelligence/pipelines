import torch
from transformers import PreTrainedTokenizer
from typing import Optional, Union, Dict, Any


class DynamicPaddingDataCollator:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        max_length: Optional[int] ="tokenizer", 
        padding: Union[bool, str] = "max_length", 
        pad_to_multiple_of: Optional[int] = None, 
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.max_length == "tokenizer":
            self.max_length = self.tokenizer.model_max_length
        
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        lengths = [len(sample["input_ids"]) for sample in batch]
        max_length = max(lengths)

        if self.max_length is not None:
            max_length = min(max_length, self.max_length)

        batch = self.tokenizer.pad(
            encoded_inputs=batch,                       
            max_length=max_length, 
            padding=self.padding, 
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            key: torch.tensor(value) if not isinstance(value[0], str) else value
            for key, value in batch.items()
        }