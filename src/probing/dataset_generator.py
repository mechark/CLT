from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class ProbingDatasetGenerator:
    """A class to generate model outputs for probing datasets."""
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model = self.model.to(device)
        self.device = device

    def _generate_batch(self, texts):
        """
        Generates model outputs for a batch of inputs.

        Args:
            texts (list): A list of input data formatted for the tokenizer.

        Returns:
            outputs: The model outputs after processing the batch.
        """
        batch = [
            [{"role": "user", "content": text}] for text in texts
        ]

        encoded = self.tokenizer.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"].to(self.device),
                attention_mask=encoded["attention_mask"].to(self.device),
                return_dict=True,
                output_hidden_states=True,
            )
        return outputs

    def get_activations(self, texts):
        """
        Retrieves hidden states.

        Args:
            texts (list): A list of input data formatted for the tokenizer.

        Returns:
            hidden_states: The hidden states.
        """

        outputs = self._generate_batch(texts)
        return outputs.hidden_states

    def store_activations(self, hidden_states, label, filepath):
        """
        Stores hidden states from a specific layer to a file.

        Args:
            layer_idx (int): The index of the layer from which to extract hidden states.
            batch (list): A list of input data formatted for the tokenizer.
            filepath (str): The path to the file where hidden states will be stored.
        """
        data = {
            "activations": {
                f"hidden_state{i}": hidden_state.cpu() for i, hidden_state in enumerate(hidden_states)
            },
            "label": label,
        }

        torch.save(data, filepath)