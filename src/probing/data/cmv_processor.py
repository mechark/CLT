import json

class CMVProcessor:
    """
    A class to process Change My View (CMV) data for probing tasks.
    """
    def __init__(self):
        self.stop_words = [
            "CMV", "Change My View", "I believe that",
            "I think that", "I feel that", "CMV: "
        ]

    def _remove_duplicates(self, corpus: list[str]) -> list[str]:
        """
        Removes duplicate texts from the corpus.

        Args:
            corpus: List of texts.
        Returns:
            List of unique texts.
        """
        return list(set(corpus))
    
    def _filter_short_texts(self, corpus: list[str], min_length: int = 20) -> list[str]:
        """
        Filters out texts that are shorter than the specified minimum length.

        Args:
            corpus: List of texts.
            min_length: Minimum length of texts to keep.
        Returns:
            List of filtered texts.
        """
        return [text for text in corpus if len(text) >= min_length]
    
    def _remove_cmv_specific_content(self, text: str) -> str:
        """
        Removes CMV-specific content such as "Delta awarded" phrases.

        Args:
            text: The input text.
        Returns:
            The cleaned text.
        """
        words = text.split()
    
    def preprocess(self, path: str, min_length: int = 20) -> list[str]:
        """
        Preprocesses the corpus by removing duplicates and filtering short texts.

        Args:
            corpus: List of texts.
            min_length: Minimum length of texts to keep.
        Returns:
            List of preprocessed texts.
        """
        with open(path, 'r') as f:
            json_strs = [json.loads(line) for line in f.readlines()]
    
        corpus = [item['submission']['title'] for item in json_strs]

        corpus = self._remove_duplicates(corpus)
        corpus = self._filter_short_texts(corpus, min_length)
        return corpus
    
if __name__ == "__main__":
    processor = CMVProcessor()
    corpus = processor.preprocess('datasets/cmv_pairs.jsonl', min_length=10)
    
    with open('datasets/cmv_processed.txt', 'w') as f:
        for text in corpus:
            f.write(text + '\n')