# model_backend.py

import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModel, AutoTokenizer

class SentimentModel:
    def __init__(self,
                 model_name_or_path: str,
                 label_map: dict,
                 max_length: int = 128,
                 weights_path: str = None):
        """
        Load tokenizer + model architecture, then optionally load saved weights.
        :param model_name_or_path: path or HuggingFace ID of base model
        :param label_map: mapping {label_name: index}
        :param max_length: max token length for inputs
        :param weights_path: (optional) path to .pth file of fine-tuned weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # tokenizer + base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,use_auth_token=True,keep_accents=True,force_download =True)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path,num_labels=len(label_map),force_download=True)
        # load weights if provided
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = label_map
        self.max_length = max_length

    def _is_malayalam(self, text: str, threshold: float = 0.6) -> bool:
        """
        Simple check if the text is predominantly Malayalam Unicode characters.
        :param text: input string
        :param threshold: proportion of Malayalam chars required
        :return: True if likely Malayalam, False otherwise
        """
        mal_chars = re.findall(r'[\u0D00-\u0D7F]', text)
        if len(text.strip()) == 0:
            return False
        proportion = len(mal_chars) / len(text)
        return proportion >= threshold

    def predict(self, text: str):
        """
        Predict sentiment of the given text if it's Malayalam.
        :param text: string input (Malayalam review)
        :return: label string or special error code
        """
        # Check for “pure Malayalam” (or at least mostly)
        if not self._is_malayalam(text):
            return "ERROR_NOT_MALAYALAM"

        # Tokenize & prepare inputs
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_index = torch.argmax(logits, dim=1).item()

        # Map index back to label
        for label, idx in self.label_map.items():
            if idx == predicted_index:
                return label

        # Fallback label
        return "UNKNOWN"
