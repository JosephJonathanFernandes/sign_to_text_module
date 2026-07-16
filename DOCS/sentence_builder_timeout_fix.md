# Fix Proposal: Disabling SentenceBuilder Auto-Sentence Timeout for WebSocket Sessions

This document outlines the proposed solution for the bug where previously recognized words (e.g. `"GOOD"`) disappear from the real-time translation stream when the user pauses before signing the next word (e.g. `"MORNING"`).

---

## 1. Root Cause Analysis

The real-time translation WebSocket (`/ws/translate` inside [api/app.py](file:///c:/DEV/Project/Final_Year/Johnny's%20Model/sign_to_text_module/api/app.py)) maintains an isolated `InferenceSession` per client. This session contains a `SentenceBuilder` instance.

By default, the `SentenceBuilder` has a built-in auto-sentence timeout set to `60` frames (approx. 2 seconds at 30 FPS):

```python
# Check for auto-sentence completion (timeout after no new words)
if self.words:
    self.frames_since_last_word += 1
    if self.frames_since_last_word >= self.auto_sentence_timeout:
        # Auto-complete and start new sentence
        raw_sentence = self.current_sentence.strip()
        # Apply NLP post-processing before storing
        completed_sentence = self.nlp_processor.process(raw_sentence, is_sentence_end=True)
        self.completed_sentences.append(completed_sentence)
        self.words.clear()  # <-- Wipes active word list
        self._rebuild_sentence()  # <-- Resets current_sentence to ""
        self.frames_since_last_word = 0
```

When this timeout triggers:
1. The active list of words is cleared.
2. The current sentence is reset to `""` and stored in `completed_sentences`.
3. The next WebSocket prediction response sends `"sentence_so_far": ""` to the client, removing the words from the frontend display.
4. Any new signs detected are added to a clean list, so they appear isolated (e.g. `"MORNING"`).
5. When the user stops the session, only the words after the last timeout are returned in the final translation summary.

---

## 2. Proposed Changes

Since the frontend client manages the translation session lifecycle (explicitly starting it, streaming frames, and stopping it), the backend should accumulate *all* words for the entire duration of the connection instead of auto-splitting.

We will implement this by allowing the timeout to be bypassed entirely when set to `0`.

### Component A: `SentenceBuilder` Update
* **File:** [src/inference/sentence_builder.py](file:///c:/DEV/Project/Final_Year/Johnny's%20Model/sign_to_text_module/src/inference/sentence_builder.py)
* **Change:** Add a check `self.auto_sentence_timeout > 0` before checking if the timeout threshold is exceeded.

```python
# Check for auto-sentence completion (timeout after no new words)
if self.auto_sentence_timeout > 0 and self.words:
    self.frames_since_last_word += 1
    if self.frames_since_last_word >= self.auto_sentence_timeout:
        # Auto-complete and start new sentence
        raw_sentence = self.current_sentence.strip()
        ...
```

### Component B: Session Configuration Update
* **File:** [api/session.py](file:///c:/DEV/Project/Final_Year/Johnny's%20Model/sign_to_text_module/api/session.py)
* **Change:** Modify the `_make_sentence_builder()` factory function to configure the `SentenceBuilder` with `auto_sentence_timeout=0` for WebSocket sessions.

```python
def _make_sentence_builder() -> SentenceBuilder:
    """Create a SentenceBuilder with production-tuned parameters."""
    return SentenceBuilder(
        confidence_threshold=0.60,
        stability_frames=8,
        ambiguity_margin_threshold=0.05,
        ambiguity_delay_frames=4,
        auto_sentence_timeout=0,  # Disable auto-clear during live API session
    )
```
