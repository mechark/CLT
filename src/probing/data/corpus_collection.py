from src.probing.data.probing_dataset import HuggingFaceDatasetLoader
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Set
from src.probing.data.cmv_processor import CMVProcessor

def extract_sentences(text: str) -> List[str]:
    """Extract all sentences from a text string.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split on sentence terminators followed by space or end of string
    sentences = re.split(r'[.!?]+\s+', text)
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def get_non_stop_words(text: str) -> Set[str]:
    """Extract non-stop words from text (simple version using common English stop words).
    
    Args:
        text: Input text
        
    Returns:
        Set of non-stop words (lowercased)
    """
    # Common English stop words
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will',
        'with', 'their', 'this', 'but', 'they', 'have', 'had', 'what', 'when', 'where',
        'who', 'which', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should',
        'now', 'or', 'been', 'were', 'would', 'could', 'there', 'any', 'also', 'may',
        'being', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here', 'his', 'her',
        'she', 'him', 'them', 'these', 'those', 'am', 'does', 'did', 'doing', 'having',
        'if', 'because', 'until', 'while', 'about', 'against', 'upon', 'out', 'over',
        'up', 'down', 'off', 'our', 'we', 'you', 'your', 'yours', 'myself', 'himself',
        'herself', 'itself', 'ourselves', 'themselves', 'i', 'me', 'my', 'mine'
    }
    
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    # Filter out stop words
    non_stop = {w for w in words if w not in stop_words and len(w) > 2}
    
    return non_stop


def calculate_word_overlap(words1: Set[str], words2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets of words.
    
    Args:
        words1: First set of words
        words2: Second set of words
        
    Returns:
        Jaccard similarity score (0-1)
    """
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def find_best_matching_sentence(
    target_words: Set[str],
    candidate_sentences: List[str],
    target_length: float,
    length_tolerance: float = 0.3
) -> Tuple[str, float]:
    """Find the best matching sentence based on word overlap and length.
    
    Args:
        target_words: Non-stop words from target text
        candidate_sentences: List of candidate sentences to match from
        target_length: Target sentence length
        length_tolerance: Allowed length deviation (0.3 = ±30%)
        
    Returns:
        Tuple of (best_sentence, overlap_score)
    """
    min_length = target_length * (1 - length_tolerance)
    max_length = target_length * (1 + length_tolerance)
    
    best_sentence = None
    best_score = 0.0
    
    for sentence in candidate_sentences:
        # Check length constraint
        if not (min_length <= len(sentence) <= max_length):
            continue
            
        # Calculate word overlap
        sentence_words = get_non_stop_words(sentence)
        overlap = calculate_word_overlap(target_words, sentence_words)
        
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence
    
    return best_sentence, best_score


def get_average_length(texts: list) -> float:
    """Calculate average character length of texts."""
    return np.mean([len(text) for text in texts])


def merge_datasets(persuasion_texts: list[str], squad_dataset, output_path: str = "./merged_dataset.csv"):
    """Merge persuasion and squad datasets with equal sampling and labels.
    
    Creates a binary classification dataset where:
    - Persuasion claims are labeled as 1
    - Squad contexts (matched sentences) are labeled as 0
    - Squad sentences are matched based on non-stop word overlap with persuasion claims
    
    Args:
        persuasion_dataset: HuggingFace dataset with 'claim' column
        squad_dataset: HuggingFace dataset with 'context' column
        output_path: Path to save the merged dataset
    """
    # Extract persuasion claims and label as 1
    persuasion_labels = [1] * len(persuasion_texts)
    n_samples = len(persuasion_texts)
    
    print(f"Persuasion dataset size: {n_samples}")
    
    # Calculate average length of persuasion texts
    avg_persuasion_length = get_average_length(persuasion_texts)
    print(f"Average persuasion text length: {avg_persuasion_length:.1f} characters")
    
    # Extract all sentences from squad contexts
    print("Extracting sentences from squad dataset...")
    squad_contexts = list(squad_dataset['context'])
    all_squad_sentences = []
    
    for ctx in squad_contexts:
        sentences = extract_sentences(ctx)
        all_squad_sentences.extend(sentences)
    
    print(f"Total squad sentences extracted: {len(all_squad_sentences)}")
    
    # For each persuasion claim, find the best matching squad sentence
    print("Matching persuasion claims with squad sentences...")
    matched_squad_sentences = []
    match_scores = []
    used_indices = set()  # Track used sentences to avoid duplicates
    
    for i, persuasion_text in enumerate(persuasion_texts):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_samples} claims...")
        
        # Get non-stop words from persuasion claim
        persuasion_words = get_non_stop_words(persuasion_text)
        
        # Find available sentences (not yet used)
        available_sentences = [
            (idx, sent) for idx, sent in enumerate(all_squad_sentences)
            if idx not in used_indices
        ]
        
        if not available_sentences:
            print(f"Warning: Ran out of unique squad sentences at claim {i}")
            # Reset used indices if we run out
            used_indices.clear()
            available_sentences = list(enumerate(all_squad_sentences))
        
        # Find best match among available sentences
        best_sentence = None
        best_score = 0.0
        best_idx = None
        
        for idx, sentence in available_sentences:
            # Check length constraint
            min_length = avg_persuasion_length * 0.7
            max_length = avg_persuasion_length * 1.3
            
            if not (min_length <= len(sentence) <= max_length):
                continue
            
            # Calculate word overlap
            sentence_words = get_non_stop_words(sentence)
            overlap = calculate_word_overlap(persuasion_words, sentence_words)
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
                best_idx = idx
        
        # If no match found with overlap, pick random sentence with correct length
        if best_sentence is None:
            fallback_candidates = [
                (idx, sent) for idx, sent in available_sentences
                if min_length <= len(sent) <= max_length
            ]
            if fallback_candidates:
                best_idx, best_sentence = fallback_candidates[
                    np.random.randint(len(fallback_candidates))
                ]
                best_score = 0.0
        
        if best_sentence:
            matched_squad_sentences.append(best_sentence)
            match_scores.append(best_score)
            used_indices.add(best_idx)
        else:
            # Last resort: pick any available sentence
            idx, sent = available_sentences[np.random.randint(len(available_sentences))]
            matched_squad_sentences.append(sent)
            match_scores.append(0.0)
            used_indices.add(idx)
    
    print(f"\nMatching complete!")
    print(f"Average match score (Jaccard similarity): {np.mean(match_scores):.3f}")
    print(f"Matches with overlap > 0: {sum(1 for s in match_scores if s > 0)} / {len(match_scores)}")
    
    squad_labels = [0] * len(matched_squad_sentences)
    
    # Combine both datasets
    all_texts = persuasion_texts + matched_squad_sentences
    all_labels = persuasion_labels + squad_labels
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    output_file = Path(output_path)
    df.to_csv(output_file, index=False)
    
    print(f"\nMerged dataset saved to: {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"\nSample matches (persuasion -> squad):")
    for i in range(min(3, len(persuasion_texts))):
        print(f"\nPersuasion (label=1): {persuasion_texts[i]}")
        print(f"Squad match (label=0): {matched_squad_sentences[i]}")
        print(f"Overlap score: {match_scores[i]:.3f}")
    
    return df


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load datasets
    loader = HuggingFaceDatasetLoader()
    
    print("Loading datasets...")
    # persuasion_dataset = loader.load_dataset("Anthropic/persuasion")
    
    df = pd.read_csv('datasets/cmv.csv')
    political_texts = df[df['class'] == 1]['statement'].tolist()

    persuasion_dataset = political_texts
    squad_dataset = loader.load_dataset("datasets/rajpurkar/squad_train")
    
    # Merge and save
    merged_df = merge_datasets(
        persuasion_dataset, 
        squad_dataset, 
        output_path="datasets/probing_dataset.csv"
    )

