from transformers import AutoTokenizer
import warnings 

warnings.simplefilter("ignore")

model_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

text = "Hello, it is simple text! This text will be tokenized."
tokenized = tokenizer(
            text=text, 
            add_special_tokens=True, 
            return_attention_mask=True, 
            padding="max_length", 
            max_length=50, 
            truncation=True, 
            return_special_tokens_mask=True, 
            return_offsets_mapping=True,
            return_token_type_ids=True,
            )

print()
print(text, end="\n"*2)
print(tokenized.offset_mapping)

print(tokenizer)