from typing import List, Optional, Dict, Any

from .dataset import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, labels: Optional[List[int]] = None, **dataset_parameters) -> None:
        super().__init__(**dataset_parameters)

        self.labels = labels

        if self.labels is not None:
            assert len(self.texts) == len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        text = self.texts[index]

        if self.texts_pair is not None:
            text_pair = self.texts_pair[index]

        tokenized = self.tokenize(text=text, text_pair=text_pair)

        if self.labels is not None:
            tokenized["label"] = int(self.labels[index])

        return tokenized