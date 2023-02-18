from typing import Dict, Any
from .classification_dataset import Dataset


class RegressionDataset(Dataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        text = self.texts[index]

        if self.texts_pair is not None:
            text_pair = self.texts_pair[index]

        tokenized = self.tokenize(text=text, text_pair=text_pair)

        if self.labels is not None:
            tokenized["label"] = float(self.labels[index])

        return tokenized