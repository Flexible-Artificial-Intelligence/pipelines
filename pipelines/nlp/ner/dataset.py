from base64 import encode
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Dict, Any

from .enums import TaggingScheme


@dataclass
class NERDataset(Dataset):
    texts: List[str]
    tokenizer: PreTrainedTokenizer
    pair_texts: Optional[List[str]] = field(default=None)
    labels: Optional[List[List[int]]] = field(default=None)
    entities: Optional[List[str]] = field(default=None)
    spans: Optional[List[List[int]]] = field(default=None)
    tagging_scheme: Optional[Union[TaggingScheme, str]] = field(default=None)
    max_length: Optional[int] = field(default=None)
    labels_key: str = field(default="labels")


    def __post_init__(self):
        if self.pair_texts is not None:
            assert  len(self.texts) == len(self.pair_texts)

        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_len


    def __getitem__(self, index: int) -> Dict[str, Any]:
        text = self.texts[index]

        pair_text = None
        if self.pair_texts is not None:
            pair_text = self.pair_texts[self.index]

        encoded = self.tokenizer(
            text=text, 
            text_pair=pair_text, 
            add_special_tokens=True,
            max_length=self.max_length,
            padding=None,
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        encoded["text"] = text

        if self.labels is not None:
            label = self.labels[index]
            encoded[self.labels_key] = label


        return encoded