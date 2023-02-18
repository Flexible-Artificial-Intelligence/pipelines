from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import List, Optional, Union, Dict, Any

from ..data_utils import get_input_length


class Dataset(Dataset):
    def __init__(
            self, 
            texts: List[str], 
            tokenizer: PreTrainedTokenizer, 
            texts_pair: Optional[List[str]] = None,
            max_length: Optional[Union[str, int]] = "input",  
            truncation: bool = True, 
            padding: bool = False, 
        ) -> None:
        super().__init__()

        self.texts = texts
        self.texts_pair = texts_pair
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

        if self.texts_pair is not None:
            assert len(self.texts) == len(self.texts_pair)
    
        if self.max_length == "input":
            lengths = []
            if self.texts_pair is not None:
                for text, text_pair in zip(self.texts, self.texts_pair):
                    length = get_input_length(text=text, text_pair=text_pair, tokenizer=self.tokenizer)
                    lengths.append(length)
            else:
                for text in self.texts:
                    length = get_input_length(text=text, tokenizer=self.tokenizer)
                    lengths.append(length)

            self.max_length = max(lengths)

        elif self.max_length == "tokenizer":
            self.max_length = self.tokenizer.model_max_length
                

    def tokenize(self, text: str, text_pair: Optional[str] = None) -> Dict[str, Any]:
        tokenized = self.tokenizer(
            text=text, 
            text_pair=text_pair,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_attention_mask=True,
            add_special_tokens=True,
            return_special_tokens_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_tensors=None,
        )
        
        return tokenized

    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        text = self.texts[index]

        if self.texts_pair is not None:
            text_pair = self.texts_pair[index]

        tokenized = self.tokenize(text=text, text_pair=text_pair)

        return tokenized