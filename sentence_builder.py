"""
Sentence builder for continuous sign language translation.

Automatically tracks predictions and builds sentences as signs are recognized.
Detects sign transitions and adds words without manual intervention.
Includes NLP post-processing (grammar, punctuation, normalization).
"""

from collections import deque
import json
from pathlib import Path
from typing import Optional, List, Tuple
from nlp_postprocessor import NLPPostProcessor


SIMILAR_SIGN_PAIRS_PATH = Path(__file__).with_name("similar_signs.json")


def _load_similar_sign_pairs() -> set[tuple[str, str]]:
    """Load similar-sign pairs from JSON."""
    try:
        with SIMILAR_SIGN_PAIRS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return set()
    except (json.JSONDecodeError, OSError):
        return set()

    raw_pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    parsed_pairs: set[tuple[str, str]] = set()

    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue

        left = str(pair[0]).strip().lower()
        right = str(pair[1]).strip().lower()
        if left and right and left != right:
            parsed_pairs.add((left, right))
            parsed_pairs.add((right, left))

    return parsed_pairs


class SentenceBuilder:
    """
    Continuously translates sign sequences into sentence text.
    
    Automatically detects when a sign completes and adds it as a word.
    Tracks confidence and stability to determine when to finalize words.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.60,
                 stability_frames: int = 8,
                 ambiguity_margin_threshold: float = 0.05,
                 ambiguity_delay_frames: int = 4,
                 auto_sentence_timeout: int = 60):
        """
        Args:
            confidence_threshold: Min confidence to consider adding a word (0-1)
            stability_frames: Frames needed to confirm sign before transition
            ambiguity_margin_threshold: Minimum top1-top2 confidence gap required
                before committing immediately
            ambiguity_delay_frames: Extra frames to wait when predictions are ambiguous
            auto_sentence_timeout: Frames before auto-completing sentence (30 frames ~= 1 sec @ 30fps)
        """
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.ambiguity_margin_threshold = ambiguity_margin_threshold
        self.ambiguity_delay_frames = ambiguity_delay_frames
        self.auto_sentence_timeout = auto_sentence_timeout
        
        # Current sign tracking
        self.current_word: Optional[str] = None
        self.current_confidence: float = 0.0
        self.stability_counter = 0
        self.last_added_word: Optional[str] = None
        self.pending_ambiguity_prediction: Optional[str] = None
        self.ambiguity_delay_counter = 0
        
        # Sentence storage
        self.words: List[str] = []
        self.current_sentence = ""
        self.completed_sentences: List[str] = []
        
        # Word history for transitions
        self.word_history = deque(maxlen=5)
        
        # Tracking
        self.transition_ready = False
        self.last_word_added_frame = 0
        self.total_frames = 0
        self.confusable_pairs = _load_similar_sign_pairs()
        
        # Auto-complete tracking
        self.frames_since_last_word = 0
        
        # ── Better Transition Logic ──
        self.prediction_history_window = deque(maxlen=10)  # Track recent predictions for smoothing
        self.idle_frames = 0  # Count frames in idle state
        self.last_transition_frame = -1000  # Frame when last word was added
        self.min_frame_gap_between_words = 5  # Minimum frames between word additions
        
        # ── NLP Post-Processing ──
        self.nlp_processor = NLPPostProcessor(
            grammar_enabled=True,
            punctuation_enabled=True,
            normalization_enabled=True
        )
        
    def update(self, 
               prediction: str, 
                             confidence: float,
                             confidence_gap: Optional[float] = None) -> dict:
        """
        Update with new frame prediction with improved transition logic.
        
        Features:
        - Smooth jittery predictions using majority voting
        - Enforce minimum frame gap between words
        - Detect genuine transitions vs noise
        - Track idle vs active signing states
        
        Returns:
            Dict with:
            - 'added_word': Added word if transition detected, else None
            - 'completed_sentence': Completed sentence if auto-timeout triggered, else None
        """
        self.total_frames += 1
        added_word = None
        completed_sentence = None
        
        # Track idle state
        if prediction == "...":
            self.idle_frames += 1
        else:
            self.idle_frames = 0
        
        # Ignore low-confidence predictions
        if confidence < self.confidence_threshold:
            prediction = "..."
        
        # Build prediction history for smoothing
        self.prediction_history_window.append((prediction, confidence))
        
        # ── Smo othed prediction using majority voting (reduces jitter) ──
        if len(self.prediction_history_window) >= 3:
            recent_preds = [p for p, _ in list(self.prediction_history_window)[-3:]]
            from collections import Counter
            smoothed_pred = Counter(recent_preds).most_common(1)[0][0]
        else:
            smoothed_pred = prediction

        if (
            confidence_gap is None or
            confidence_gap >= self.ambiguity_margin_threshold or
            self.ambiguity_delay_frames == 0
        ):
            self.pending_ambiguity_prediction = None
            self.ambiguity_delay_counter = 0
        
        # Check for auto-sentence completion (timeout after no new words)
        if self.words:
            self.frames_since_last_word += 1
            if self.frames_since_last_word >= self.auto_sentence_timeout:
                # Auto-complete and start new sentence
                raw_sentence = self.current_sentence.strip()
                # Apply NLP post-processing before storing
                completed_sentence = self.nlp_processor.process(raw_sentence, is_sentence_end=True)
                self.completed_sentences.append(completed_sentence)
                self.words.clear()
                self._rebuild_sentence()
                self.frames_since_last_word = 0
        
        # ── Improved Transition Logic ──
        # Use smoothed prediction for transition detection
        if smoothed_pred != self.current_word:
            # Ignore transitions from/to idle unless very certain
            if smoothed_pred == "..." or self.current_word == "...":
                # Transitioning from/to idle: require more stability
                min_stability = max(self.stability_frames + 2, 10)
            else:
                # Word-to-word transition: normal stability
                min_stability = self.stability_frames

            is_ambiguous_transition = (
                confidence_gap is not None and
                confidence_gap < self.ambiguity_margin_threshold and
                self.ambiguity_delay_frames > 0
            )
            
            # Finalize previous sign if stable and genuinely different
            if (self.current_word is not None and 
                self.current_word != "..." and 
                self.stability_counter >= min_stability and
                self.current_word != self.last_added_word):

                if is_ambiguous_transition:
                    if self.pending_ambiguity_prediction != smoothed_pred:
                        self.pending_ambiguity_prediction = smoothed_pred
                        self.ambiguity_delay_counter = 0

                    self.ambiguity_delay_counter += 1
                    if self.ambiguity_delay_counter < self.ambiguity_delay_frames:
                        self.transition_ready = False
                        self.current_confidence = max(self.current_confidence, confidence)
                        return {
                            'added_word': added_word,
                            'completed_sentence': completed_sentence,
                        }
                
                # Enforce minimum frame gap between words
                frames_since_last = self.total_frames - self.last_word_added_frame
                if frames_since_last >= self.min_frame_gap_between_words:
                    added_word = self._add_word(self.current_word)
                    self.last_added_word = self.current_word
                    if added_word:
                        self.last_word_added_frame = self.total_frames
                        self.frames_since_last_word = 0
                        self.last_transition_frame = self.total_frames
            
            # Start tracking new sign
            self.current_word = smoothed_pred
            self.current_confidence = confidence
            self.stability_counter = 1
            self.transition_ready = False
            self.pending_ambiguity_prediction = None
            self.ambiguity_delay_counter = 0
            
        else:
            # Same prediction, increase stability
            self.stability_counter += 1
            self.current_confidence = max(
                self.current_confidence, confidence
            )
            self.pending_ambiguity_prediction = None
            self.ambiguity_delay_counter = 0
            
            # Mark as ready to transition when stable enough
            if self.stability_counter >= self.stability_frames:
                self.transition_ready = True
        
        return {
            'added_word': added_word,
            'completed_sentence': completed_sentence
        }
    
    
    def _add_word(self, word: str) -> str:
        """Internal: add word to sentence."""
        if not word or word == "...":
            return ""
        
        # Skip if duplicate of last word
        if self.words and self.words[-1].lower() == word.lower():
            return ""
        
        self.words.append(word.upper())
        self.word_history.append(word.upper())
        self._rebuild_sentence()
        return word.upper()
    
    def _rebuild_sentence(self) -> None:
        """Rebuild full sentence text from word list."""
        self.current_sentence = " ".join(self.words)
    
    def is_confusable_pair(self, word1: Optional[str], word2: Optional[str]) -> bool:
        """Check if two words are easily confused (similar signs).
        
        These pairs typically need stricter thresholds during transitions.
        """
        if not word1 or not word2 or word1 == word2:
            return False
        
        pair = (word1.lower(), word2.lower())
        return pair in self.confusable_pairs
    
    def get_transition_requirement(self, prev_word: Optional[str], next_word: Optional[str]) -> float:
        """
        Get confidence requirement multiplier for transition between words.
        
        Args:
            prev_word: Previous word (can be None or "...")
            next_word: Next word being considered
            
        Returns:
            Multiplier to apply to base threshold (1.0 = normal, 1.2 = 20% stricter)
        """
        if not prev_word or prev_word == "...":
            return 1.0  # Starting new sign - normal threshold
        
        if not next_word or next_word == "...":
            return 1.2  # Transitioning to idle - stricter (avoid false negatives)
        
        if self.is_confusable_pair(prev_word, next_word):
            return 1.3  # Very strict for confusable pairs
        
        return 1.0  # Normal word-to-word transition
    
    def undo_word(self) -> Optional[str]:
        """Remove last word from sentence."""
        if not self.words:
            return None
        
        removed = self.words.pop()
        self._rebuild_sentence()
        return removed
    
    def add_punctuation(self, punct: str) -> str:
        """
        Add punctuation to sentence.
        
        Args:
            punct: Punctuation mark ('.' for period, '!' for exclamation, etc.)
            
        Returns:
            Updated sentence text
        """
        if self.current_sentence:
            self.current_sentence += punct
        elif self.words:
            self.words[-1] = self.words[-1] + punct
            self._rebuild_sentence()
        return self.current_sentence

    def flush_pending_word(self) -> Optional[str]:
        """Force-add the current tracked word if one is still pending."""
        if not self.current_word or self.current_word == "...":
            return None

        pending = self.current_word
        if self.current_word != self.last_added_word:
            added = self._add_word(self.current_word)
            if added:
                self.last_added_word = self.current_word
                self.frames_since_last_word = 0
                return added

        return pending.upper()
    
    def clear(self) -> str:
        """Clear entire sentence."""
        self.words.clear()
        self.current_sentence = ""
        self._rebuild_sentence()
        return ""
    
    def save_sentence(self) -> str:
        """
        Finalize and return complete sentence with post-processing.
        
        Returns:
            Post-processed sentence string
        """
        sentence = self.current_sentence.strip()
        # Apply NLP post-processing
        processed = self.nlp_processor.process(sentence, is_sentence_end=True)
        return processed
    
    def get_stats(self) -> dict:
        """Get debugging stats."""
        nlp_stats = self.nlp_processor.get_stats()
        return {
            'current_word': self.current_word,
            'stability': f"{self.stability_counter}/{self.stability_frames}",
            'confidence': f"{self.current_confidence:.1%}",
            'ambiguity_delay': f"{self.ambiguity_delay_counter}/{self.ambiguity_delay_frames}",
            'word_count': len(self.words),
            'sentence_length': len(self.current_sentence),
            'ready_to_transition': self.transition_ready,
            'nlp_processing': nlp_stats,
        }
    
    def get_display_text(self) -> dict:
        """
        Get formatted text for display.
        
        Returns:
            Dict with 'words' list and 'sentence' string
        """
        stability_pct = min(100, (self.stability_counter / self.stability_frames) * 100)
        return {
            'words': self.words.copy(),
            'sentence': self.current_sentence,
            'current': f"{self.current_word}" if self.current_word and self.current_word != "..." else "—",
            'stability': self.stability_counter,
            'stability_max': self.stability_frames,
            'confidence': f"{self.current_confidence:.0%}",
            'ambiguity_delay': self.ambiguity_delay_counter,
            'ambiguity_delay_max': self.ambiguity_delay_frames,
            'ready': self.transition_ready,
        }
    
    def set_nlp_grammar_enabled(self, enabled: bool) -> None:
        """Enable/disable grammar correction."""
        self.nlp_processor.grammar_corrector.enabled = enabled
    
    def set_nlp_punctuation_enabled(self, enabled: bool) -> None:
        """Enable/disable automatic punctuation insertion."""
        self.nlp_processor.punctuation_inserter.enabled = enabled
    
    def set_nlp_normalization_enabled(self, enabled: bool) -> None:
        """Enable/disable text normalization."""
        self.nlp_processor.text_normalizer.enabled = enabled
    
    def get_nlp_status(self) -> dict:
        """Get current NLP post-processing status."""
        return self.nlp_processor.get_stats()
    
    def preprocess_text(self, text: str, is_sentence_end: bool = False) -> str:
        """
        Apply NLP post-processing to raw text (useful for manual input).
        
        Args:
            text: Raw text to process
            is_sentence_end: Whether to insert punctuation
        
        Returns:
            Processed text
        """
        return self.nlp_processor.process(text, is_sentence_end=is_sentence_end)


class SentenceEditor:
    """
    Manages multiple sentences and editing operations.
    """
    
    def __init__(self):
        self.current_builder = SentenceBuilder()
        self.completed_sentences: List[str] = []
        
    def new_sentence(self) -> None:
        """Start a new sentence."""
        if self.current_builder.current_sentence:
            self.save_current()
        self.current_builder.clear()
    
    def save_current(self) -> str:
        """Save current sentence to history."""
        sentence = self.current_builder.save_sentence()
        if sentence:
            self.completed_sentences.append(sentence)
        return sentence
    
    def get_all_text(self, sep: str = " ") -> str:
        """Get all completed text."""
        text = sep.join(self.completed_sentences)
        if self.current_builder.current_sentence:
            text += sep + self.current_builder.current_sentence
        return text.strip()
    
    def export(self) -> str:
        """Export all text."""
        return self.get_all_text(sep=". ").rstrip() + "."
