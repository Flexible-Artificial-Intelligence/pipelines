from transformers import PreTrainedTokenizer
from typing import Optional


def get_input_length(
        text: str, 
        tokenizer: PreTrainedTokenizer, 
        text_pair: Optional[str] = None,
    ) -> int:
    tokenized = tokenizer(
        text=text, 
        text_pair=text_pair, 
        add_special_tokens=True, 
        return_attention_mask=False,
    )

    input_ids = tokenized["input_ids"]
    length = len(input_ids)
    
    return length