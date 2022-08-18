from typing import List, Any, Union

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