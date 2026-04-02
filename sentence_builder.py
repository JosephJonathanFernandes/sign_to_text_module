"""
Sentence builder for continuous sign language translation.

Automatically tracks predictions and builds sentences as signs are recognized.
Detects sign transitions and adds words without manual intervention.
"""

from collections import deque
from typing import Optional, List, Tuple


class SentenceBuilder:
    """
    Continuously translates sign sequences into sentence text.
    
    Automatically detects when a sign completes and adds it as a word.
    Tracks confidence and stability to determine when to finalize words.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.60,
                 stability_frames: int = 8,
                 auto_sentence_timeout: int = 60):
        """
        Args:
            confidence_threshold: Min confidence to consider adding a word (0-1)
            stability_frames: Frames needed to confirm sign before transition
            auto_sentence_timeout: Frames before auto-completing sentence (30 frames ~= 1 sec @ 30fps)
        """
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.auto_sentence_timeout = auto_sentence_timeout
        
        # Current sign tracking
        self.current_word: Optional[str] = None
        self.current_confidence: float = 0.0
        self.stability_counter = 0
        self.last_added_word: Optional[str] = None
        
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
        
        # Auto-complete tracking
        self.frames_since_last_word = 0
        
    def update(self, 
               prediction: str, 
               confidence: float) -> dict:
        """
        Update with new frame prediction.
        
        Returns:
            Dict with:
            - 'added_word': Added word if transition detected, else None
            - 'completed_sentence': Completed sentence if auto-timeout triggered, else None
        """
        self.total_frames += 1
        added_word = None
        completed_sentence = None
        
        # Ignore low-confidence predictions
        if confidence < self.confidence_threshold:
            prediction = "..."
        
        # Check for auto-sentence completion (timeout after no new words)
        if self.words:
            self.frames_since_last_word += 1
            if self.frames_since_last_word >= self.auto_sentence_timeout:
                # Auto-complete and start new sentence
                completed_sentence = self.current_sentence.strip()
                self.completed_sentences.append(completed_sentence)
                self.words.clear()
                self._rebuild_sentence()
                self.frames_since_last_word = 0
        
        # New prediction detected
        if prediction != self.current_word:
            # Finalize previous sign if it was stable and different
            if (self.current_word is not None and 
                self.current_word != "..." and 
                self.stability_counter >= self.stability_frames and
                self.current_word != self.last_added_word):
                
                added_word = self._add_word(self.current_word)
                self.last_added_word = self.current_word
                if added_word:
                    self.last_word_added_frame = self.total_frames
                    self.frames_since_last_word = 0
            
            # Start tracking new sign
            self.current_word = prediction
            self.current_confidence = confidence
            self.stability_counter = 1
            self.transition_ready = False
            
        else:
            # Same prediction, increase stability
            self.stability_counter += 1
            self.current_confidence = max(
                self.current_confidence, confidence
            )
            
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
    
    def clear(self) -> str:
        """Clear entire sentence."""
        self.words.clear()
        self.current_sentence = ""
        self._rebuild_sentence()
        return ""
    
    def save_sentence(self) -> str:
        """
        Finalize and return complete sentence.
        
        Returns:
            Complete sentence string
        """
        return self.current_sentence.strip()
    
    def get_stats(self) -> dict:
        """Get debugging stats."""
        return {
            'current_word': self.current_word,
            'stability': f"{self.stability_counter}/{self.stability_frames}",
            'confidence': f"{self.current_confidence:.1%}",
            'word_count': len(self.words),
            'sentence_length': len(self.current_sentence),
            'ready_to_transition': self.transition_ready,
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
            'ready': self.transition_ready,
        }


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
