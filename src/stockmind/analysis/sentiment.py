"""FinBERT sentiment for news headlines, scored once and remembered.

FinBERT (ProsusAI/finbert; Araci, 2019) is BERT trained further on
financial text and fine-tuned on Financial PhraseBank, sentences from
financial news labelled positive, negative or neutral by finance experts.
General sentiment models get financial language wrong ("costs fell" is
good news), which is why the proposal uses FinBERT. It is used as
published: no training is needed.

For each text the model gives three probabilities that add up to 1. The
sentiment of a text is P(positive) - P(negative), from -1 to +1.

Speed
    Loading the model takes a few seconds and scoring a headline a few
    milliseconds on a normal CPU. Every score is remembered (in
    ~/.stockmind/cache/sentiment, keyed by model and text), so each
    headline is analysed once and later runs do not even load the model.
    Like the rest of the cache, the folder can be deleted at any time.

PyTorch and Transformers are imported only when the model is first needed,
so importing this module (and running the tests) stays fast.
"""

import hashlib
import logging
import re
from pathlib import Path

import diskcache
import numpy as np
import pandas as pd

from stockmind import config

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ["positive", "negative", "neutral", "sentiment", "sentiment_label"]
LABELS = ["positive", "negative", "neutral"]


class ModelUnavailableError(RuntimeError):
    """FinBERT could not be loaded (not installed, or not downloadable)."""


class FinBERT:
    """The FinBERT model, loaded the first time it is used."""

    def __init__(
        self,
        model_name: str = config.SENTIMENT_MODEL,
        revision: str = config.SENTIMENT_MODEL_REVISION,
        batch_size: int = config.SENTIMENT_BATCH_SIZE,
        max_tokens: int = config.SENTIMENT_MAX_TOKENS,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._tokenizer = self._model = self._order = None

    @property
    def name(self) -> str:
        """Identifies the exact model, so a new revision gets new scores."""
        return f"{self.model_name}@{self.revision[:12]}"

    def _load(self) -> None:
        try:
            import torch  # noqa: F401  (imported here: slow to import)
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            from transformers.utils import logging as transformers_logging
            transformers_logging.disable_progress_bar()   # no "Loading weights" bar
        except ImportError as exc:
            raise ModelUnavailableError(
                "PyTorch and Transformers are not installed "
                "(pip install -r requirements.txt)") from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.revision).eval()
        except OSError as exc:  # no connection on the first download, etc.
            raise ModelUnavailableError(
                f"{self.model_name} could not be loaded ({exc})") from exc
        # The model's own label order, e.g. {0: "positive", 1: "negative", ...}.
        id2label = {i: label.lower() for i, label in self._model.config.id2label.items()}
        self._order = [next(i for i, label in id2label.items() if label == wanted)
                       for wanted in LABELS]

    def classify(self, texts: list[str]) -> np.ndarray:
        """P(positive), P(negative), P(neutral) for each text (rows)."""
        if not texts:
            return np.empty((0, len(LABELS)))
        if self._model is None:
            self._load()
        import torch

        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = self._tokenizer(
                    texts[start:start + self.batch_size], padding=True,
                    truncation=True, max_length=self.max_tokens, return_tensors="pt")
                probabilities = torch.softmax(self._model(**tokens).logits, dim=-1)
                batches.append(probabilities[:, self._order].numpy())
        return np.vstack(batches)


def default_score_dir() -> Path:
    return Path(config.CACHE_DIR) / "sentiment"


# Pakistani newspapers write "53pc" for "53 percent". FinBERT, trained on
# international financial news, does not know "pc": "OGDC profit surges 53pc
# and declares dividend" scored -0.25, and +0.61 with "percent".
_PERCENT = re.compile(r"(?<=\d)\s?pc\b", re.IGNORECASE)


def normalise(text: str) -> str:
    """
    The text FinBERT reads. FinBERT ignores case, so "DIVIDEND" and
    "Dividend" share one score; "pc" after a number becomes "percent".
    """
    text = _PERCENT.sub(" percent", str(text))
    return " ".join(text.split()).lower()


class SentimentScorer:
    """Score texts with FinBERT, remembering every score."""

    def __init__(self, model=None, directory: Path | str | None = None) -> None:
        self._model = model or FinBERT()
        self._directory = Path(directory or default_score_dir())
        self._store = None  # opened on first use

    def _key(self, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{self._model.name}:{digest}"

    def score(self, texts) -> tuple[pd.DataFrame, list[str]]:
        """
        Sentiment of each text, in the same order (SCORE_COLUMNS).

        Never raises for an unavailable model: texts scored before still get
        their remembered scores, the others are left empty, and a warning
        explains why.
        """
        texts = [normalise(text) for text in texts]
        if self._store is None:
            self._store = diskcache.Cache(str(self._directory))
        # dict.fromkeys: each text once, in its original order.
        known = {text: self._store.get(self._key(text)) for text in dict.fromkeys(texts)}
        new = [text for text, scores in known.items() if scores is None]

        warnings = []
        if new:
            try:
                probabilities = self._model.classify(new)
            except ModelUnavailableError as exc:
                warnings.append(f"FinBERT is unavailable, so {len(new)} news "
                                f"items were not analysed: {exc}")
            else:
                for text, row in zip(new, probabilities):
                    known[text] = tuple(float(p) for p in row)
                    self._store.set(self._key(text), known[text])
                logger.info("FinBERT scored %d new texts", len(new))

        rows = [known[text] or (np.nan, np.nan, np.nan) for text in texts]
        table = pd.DataFrame(rows, columns=LABELS)
        table["sentiment"] = table["positive"] - table["negative"]
        scored = table["positive"].notna()
        table["sentiment_label"] = None
        if scored.any():
            table.loc[scored, "sentiment_label"] = table.loc[scored, LABELS].idxmax(axis=1)
        return table[SCORE_COLUMNS], warnings
