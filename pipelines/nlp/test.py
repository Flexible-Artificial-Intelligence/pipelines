import transformers
from transformers import AutoTokenizer


print(transformers.__version__)
# 4.21.1

# Config
MAX_LENGTH = 25
MODEL_PATH = "bert-base-uncased"

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# define batch of texts
texts = ["HuggingFace Transformers is great library", "We need to fix this issue :)."]

# encode texts and send to batch
encoded_inputs = []
for text in texts:
    encoded_input = tokenizer(
        text=text, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        return_offsets_mapping=True,
        padding="do_not_pad",
        truncation=True,
        return_token_type_ids=False,
    )

    encoded_inputs.append(encoded_input)

print(encoded_inputs, end="\n"*3)

# [
#     {
#         'input_ids': [101, 17662, 12172, 19081, 2003, 2307, 3075, 102], 
#         'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1], 
#         'offset_mapping': [(0, 0), (0, 7), (7, 11), (12, 24), (25, 27), (28, 33), (34, 41), (0, 0)]
#     }, 
#     {
#         'input_ids': [101, 2057, 2342, 2000, 8081, 2023, 3277, 1024, 1007, 1012, 102], 
#         'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#         'offset_mapping': [(0, 0), (0, 2), (3, 7), (8, 10), (11, 14), (15, 19), (20, 25), (26, 27), (27, 28), (28, 29), (0, 0)]
#     }
# ]


# Dynamic Padding
padded_encoded_inputs = tokenizer.pad(
    encoded_inputs=encoded_inputs, 
    padding="max_length", 
    max_length=MAX_LENGTH,
)

print(padded_encoded_inputs)

# {
#     'input_ids': [[101, 17662, 12172, 19081, 2003, 2307, 3075, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2057, 2342, 2000, 8081, 2023, 3277, 1024, 1007, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
#     'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
#     'offset_mapping': [[(0, 0), (0, 7), (7, 11), (12, 24), (25, 27), (28, 33), (34, 41), (0, 0)], [(0, 0), (0, 2), (3, 7), (8, 10), (11, 14), (15, 19), (20, 25), (26, 27), (27, 28), (28, 29), (0, 0)]]
# }