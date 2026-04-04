"""
NLP Post-processing for ISL Sign Language Text.

Includes:
- Grammar correction (subject-verb agreement, articles, tense)
- Punctuation insertion (automatic sentence end detection)
- Text normalization (capitalization, whitespace, duplicates)

All lightweight, zero external dependencies.
"""

import re
from typing import Optional, List, Tuple


class GrammarCorrector:
    """
    Rule-based grammar correction for ISL translation output.
    
    Fixes common patterns that arise from direct sign→text translation:
    - Missing articles (a, an, the)
    - Subject-verb agreement
    - Pronoun cases
    - Tense normalization
    - Verb forms
    """
    
    # Common words that typically need articles
    COUNTABLE_WORDS = {
        'boy', 'girl', 'man', 'woman', 'person', 'child', 'cat', 'dog', 'bird',
        'book', 'table', 'chair', 'door', 'window', 'house', 'car', 'apple',
        'hand', 'face', 'eye', 'ear', 'head', 'friend', 'teacher', 'student',
    }
    
    # Words that typically follow with/without articles
    UNCOUNTABLE_WORDS = {
        'water', 'milk', 'coffee', 'tea', 'juice', 'rice', 'bread', 'butter',
        'information', 'knowledge', 'furniture', 'luggage', 'baggage', 'money',
    }
    
    # Subject-verb agreement rules (singular verbs)
    SINGULAR_VERBS = {
        'is', 'has', 'goes', 'does', 'sees', 'likes', 'loves', 'hates',
        'wants', 'needs', 'gives', 'takes', 'makes', 'comes', 'stays',
        'eats', 'drinks', 'sleeps', 'walks', 'runs', 'plays', 'works',
        'helps', 'learns', 'reads', 'watches', 'writes', 'teaches',
    }
    
    # Plural verb forms
    PLURAL_VERBS = {
        'are', 'have', 'go', 'do', 'see', 'like', 'love', 'hate',
        'want', 'need', 'give', 'take', 'make', 'come', 'stay',
        'eat', 'drink', 'sleep', 'walk', 'run', 'play', 'work',
        'help', 'learn', 'read', 'watch', 'write', 'teach',
    }
    
    # Singular pronouns
    SINGULAR_PRONOUNS = {'i', 'he', 'she', 'it'}
    
    # Plural pronouns
    PLURAL_PRONOUNS = {'we', 'they', 'you'}
    
    # Verb lemmas and their forms
    VERB_FORMS = {
        'go': {'present': 'go', 'goes': 'go', 'went': 'go', 'past': 'went'},
        'see': {'present': 'see', 'sees': 'see', 'saw': 'see', 'past': 'saw'},
        'give': {'present': 'give', 'gives': 'give', 'gave': 'give', 'past': 'gave'},
        'take': {'present': 'take', 'takes': 'take', 'took': 'take', 'past': 'took'},
        'make': {'present': 'make', 'makes': 'make', 'made': 'make', 'past': 'made'},
        'come': {'present': 'come', 'comes': 'come', 'came': 'come', 'past': 'came'},
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def correct(self, text: str) -> str:
        """Apply grammar corrections to text."""
        if not self.enabled or not text:
            return text
        
        words = text.lower().split()
        
        # Fix subject-verb agreement
        words = self._fix_subject_verb_agreement(words)
        
        # Fix missing articles
        words = self._fix_articles(words)
        
        # Restore original capitalization (first word)
        if words:
            words[0] = words[0].capitalize()
        
        return " ".join(words)
    
    def _fix_subject_verb_agreement(self, words: List[str]) -> List[str]:
        """Ensure subject-verb agreement in sentences."""
        corrected = words.copy()
        
        # Mapping of plural verbs to singular forms
        verb_map = {
            'go': 'goes', 'do': 'does', 'see': 'sees', 'like': 'likes',
            'love': 'loves', 'hate': 'hates', 'want': 'wants', 'need': 'needs',
            'give': 'gives', 'take': 'takes', 'make': 'makes', 'come': 'comes',
            'stay': 'stays', 'eat': 'eats', 'drink': 'drinks', 'sleep': 'sleeps',
            'walk': 'walks', 'run': 'runs', 'play': 'plays', 'work': 'works',
            'help': 'helps', 'learn': 'learns', 'read': 'reads', 'watch': 'watches',
            'write': 'writes', 'teach': 'teaches', 'have': 'has',
            'are': 'is',  # Special handling for be
        }
        
        for i in range(len(corrected) - 1):
            current = corrected[i].lower()
            next_word = corrected[i + 1].lower()
            
            # Check if current is a singular pronoun
            if current in self.SINGULAR_PRONOUNS:
                # i, he, she, it → singular forms
                if next_word in verb_map:
                    corrected[i + 1] = verb_map[next_word]
                elif next_word == 'am' and current != 'i':
                    # am → is for he/she/it
                    corrected[i + 1] = 'is'
            
            elif current in self.PLURAL_PRONOUNS or current in {'people', 'children', 'men', 'women'}:
                # we, they, you → plural forms
                reverse_map = {v: k for k, v in verb_map.items()}
                if next_word in reverse_map:
                    corrected[i + 1] = reverse_map[next_word]
                elif next_word == 'is':
                    # is → are for plural
                    corrected[i + 1] = 'are'
        
        return corrected
    
    def _fix_articles(self, words: List[str]) -> List[str]:
        """Insert missing articles (a, an, the) where appropriate."""
        corrected = []
        
        for i, word in enumerate(words):
            # Skip if word already has article
            if word in {'a', 'an', 'the'}:
                corrected.append(word)
                continue
            
            # Check if noun needs article
            if word in self.COUNTABLE_WORDS:
                # Look back for subject
                prev_word = words[i - 1].lower() if i > 0 else None
                
                # Add article if missing
                if prev_word not in {'a', 'an', 'the'} and prev_word not in {'is', 'are', 'have', 'has'}:
                    article = 'an' if word[0] in 'aeiou' else 'a'
                    corrected.append(article)
            
            corrected.append(word)
        
        return corrected
    
    def fix_common_patterns(self, text: str) -> str:
        """Fix common ISL→English translation patterns."""
        if not self.enabled:
            return text
        
        # Fix doubled words in sequence
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        # Fix common sign language artifacts
        patterns = [
            (r'\bi\s+am\b', 'i am'),  # Normalize spacing
            (r'\bhe\s+go\b', 'he goes'),
            (r'\bshe\s+go\b', 'she goes'),
            (r'\bit\s+is\b', 'it is'),
            (r'\byou\s+is\b', 'you are'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class PunctuationInserter:
    """
    Heuristic-based punctuation insertion for sign language translation.
    
    Detects natural sentence boundaries and inserts appropriate punctuation:
    - Period (.) for statement endings
    - Question mark (?) for questions
    - Exclamation mark (!) for emphatic signs
    """
    
    # Words that typically start questions
    QUESTION_STARTERS = {
        'who', 'what', 'where', 'when', 'why', 'how', 'which', 'do', 'does',
        'is', 'are', 'can', 'could', 'will', 'would', 'should', 'have', 'has',
    }
    
    # Words that typically indicate questions mid-sentence
    QUESTION_WORDS = (QUESTION_STARTERS | {'ask', 'question'})
    
    # Words that typically indicate emphatic/exclamatory signs
    EMPHATIC_WORDS = {
        'love', 'hate', 'beautiful', 'ugly', 'amazing', 'terrible', 'wonderful',
        'terrible', 'fantastic', 'horrible', 'excellent', 'perfect', 'awful',
    }
    
    # Pauses/fillers in ISL
    PAUSE_WORDS = {'pause', 'wait', 'stop', 'hold', 'freeze'}
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.last_was_question = False
        self.consecutive_emphatic = 0
    
    def insert(self, text: str, is_sentence_end: bool = True) -> str:
        """
        Insert punctuation into text.
        
        Args:
            text: Text to punctuate
            is_sentence_end: Whether this signals the end of a sentence
        
        Returns:
            Text with appropriate punctuation
        """
        if not self.enabled or not text or text.endswith(('.', '!', '?')):
            return text
        
        # Skip punctuation insertion if no ending signal
        if not is_sentence_end:
            return text
        
        punctuation = self._determine_punctuation(text)
        return text + punctuation
    
    def _determine_punctuation(self, text: str) -> str:
        """Determine appropriate punctuation for text."""
        words = text.lower().split()
        if not words:
            return '.'
        
        first_word = words[0]
        
        # Question detection
        if first_word in self.QUESTION_STARTERS:
            self.last_was_question = True
            return '?'
        
        # Check for question-type content mid-sentence
        if any(word in self.QUESTION_WORDS for word in words):
            if 'ask' in words or 'question' in words:
                return '?'
        
        # Emphatic detection
        emphatic_count = sum(1 for word in words if word in self.EMPHATIC_WORDS)
        if emphatic_count >= 1:
            self.consecutive_emphatic += 1
            if self.consecutive_emphatic >= 2:
                self.consecutive_emphatic = 0
                return '!'
        else:
            self.consecutive_emphatic = 0
        
        # Multiple emphatic words → exclamation
        if emphatic_count >= 2:
            return '!'
        
        # Default to period
        return '.'
    
    def suggest_punctuation(self, text: str) -> Tuple[str, str]:
        """
        Suggest punctuation and reasoning.
        
        Returns:
            Tuple of (punctuation, reason)
        """
        words = text.lower().split()
        
        # Check for questions
        if words[0] in self.QUESTION_STARTERS:
            return '?', 'Question detected'
        
        # Check for emphatic
        emphatic = sum(1 for w in words if w in self.EMPHATIC_WORDS)
        if emphatic >= 2:
            return '!', 'Multiple emphatic words'
        
        return '.', 'Default statement'


class TextNormalizer:
    """
    Normalize text output from sign language translation.
    
    Handles:
    - Capitalization
    - Whitespace cleanup
    - Duplicate word removal
    - Abbreviation expansion
    - Spacing around punctuation
    """
    
    # Abbreviations to expand
    ABBREVIATIONS = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "haven't": "have not",
        "hasn't": "has not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "it's": "it is",
        "i'm": "i am",
        "we're": "we are",
        "they're": "they are",
        "you're": "you are",
    }
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def normalize(self, text: str) -> str:
        """Apply all normalizations."""
        if not self.enabled or not text:
            return text
        
        # Clean whitespace
        text = self._clean_whitespace(text)
        
        # Remove duplicate words
        text = self._remove_duplicates(text)
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Normalize capitalization
        text = self._normalize_capitalization(text)
        
        # Fix spacing around punctuation
        text = self._fix_punctuation_spacing(text)
        
        return text.strip()
    
    def _clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace and newlines."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _remove_duplicates(self, text: str) -> str:
        """Remove consecutive duplicate words."""
        words = text.split()
        normalized = []
        
        for i, word in enumerate(words):
            # Skip if same as previous (case-insensitive)
            if i > 0 and word.lower() == words[i - 1].lower():
                continue
            normalized.append(word)
        
        return ' '.join(normalized)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common contractions."""
        for abbrev, expanded in self.ABBREVIATIONS.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbrev), re.IGNORECASE)
            text = pattern.sub(expanded, text)
        
        return text
    
    def _normalize_capitalization(self, text: str) -> str:
        """
        Normalize capitalization:
        - First word capitalized
        - I always capitalized
        - Other words lowercase (except proper nouns - basic heuristic)
        """
        words = text.split()
        
        for i, word in enumerate(words):
            # Always capitalize 'I'
            if word.lower() == 'i':
                words[i] = 'I'
            # Capitalize first word
            elif i == 0:
                words[i] = word.capitalize()
            # Lowercase others (except after sentence breaks)
            elif not word[0].isupper() or word.lower() in ['a', 'an', 'the', 'is', 'are']:
                words[i] = word.lower()
        
        return ' '.join(words)
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation marks."""
        # No space before punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        # Space after punctuation (but not multiple)
        text = re.sub(r'([.!?,;:])\s*', r'\1 ', text)
        
        # Clean up multiple spaces that might have been added
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class NLPPostProcessor:
    """
    Main post-processor combining all NLP components.
    
    Orchestrates grammar correction, punctuation insertion, and text normalization.
    """
    
    def __init__(self, 
                 grammar_enabled: bool = True,
                 punctuation_enabled: bool = True,
                 normalization_enabled: bool = True):
        """
        Initialize post-processor.
        
        Args:
            grammar_enabled: Enable grammar correction
            punctuation_enabled: Enable automatic punctuation
            normalization_enabled: Enable text normalization
        """
        self.grammar_corrector = GrammarCorrector(enabled=grammar_enabled)
        self.punctuation_inserter = PunctuationInserter(enabled=punctuation_enabled)
        self.text_normalizer = TextNormalizer(enabled=normalization_enabled)
    
    def process(self, text: str, is_sentence_end: bool = False) -> str:
        """
        Process text through all post-processing steps.
        
        Args:
            text: Raw ISL translation text
            is_sentence_end: Whether this marks the end of a sentence
        
        Returns:
            Post-processed, corrected text
        """
        if not text:
            return text
        
        # Step 1: Fix common grammar patterns first
        text = self.grammar_corrector.correct(text)
        text = self.grammar_corrector.fix_common_patterns(text)
        
        # Step 2: Add punctuation if sentence end
        if is_sentence_end:
            text = self.punctuation_inserter.insert(text, is_sentence_end=True)
        
        # Step 3: Normalize text
        text = self.text_normalizer.normalize(text)
        
        return text
    
    def get_stats(self) -> dict:
        """Get post-processor status."""
        return {
            'grammar_correction': self.grammar_corrector.enabled,
            'punctuation_insertion': self.punctuation_inserter.enabled,
            'text_normalization': self.text_normalizer.enabled,
        }


# Example usage and testing
if __name__ == '__main__':
    processor = NLPPostProcessor()
    
    test_cases = [
        ("hello world", False),
        ("she go home", True),
        ("what you name", True),
        ("i love this", True),
        ("you is happy", True),
        ("he  go home", True),  # Duplicate 'go'
    ]
    
    print("=== NLP Post-Processor Test ===\n")
    for text, is_end in test_cases:
        result = processor.process(text, is_sentence_end=is_end)
        print(f"Input:  '{text}' (end={is_end})")
        print(f"Output: '{result}'")
        print()
