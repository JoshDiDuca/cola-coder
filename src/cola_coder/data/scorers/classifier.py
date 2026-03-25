"""Distilled quality classifier — train from LLM annotations, fast inference."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from cola_coder.data.scorers.protocol import ScorerResult


@dataclass
class ClassifierMetrics:
    """Metrics from classifier training/evaluation."""
    accuracy: float = 0.0
    mean_absolute_error: float = 0.0
    num_train: int = 0
    num_test: int = 0
    per_class: dict[int, float] = field(default_factory=dict)


class QualityClassifierTrainer:
    """Train a fast quality classifier from LLM annotations."""

    def train(
        self,
        annotations_path: str,
        output_dir: str,
        test_fraction: float = 0.1,
    ) -> ClassifierMetrics:
        """Train TF-IDF + logistic regression classifier.

        Args:
            annotations_path: Path to annotations.jsonl from LlmJudge.annotate_batch.
            output_dir: Directory to save vectorizer.pkl + model.pkl.
            test_fraction: Fraction of data to hold out for evaluation.

        Returns:
            ClassifierMetrics with accuracy and error metrics.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, mean_absolute_error
        except ImportError:
            raise ImportError(
                "scikit-learn is required for classifier training. "
                "Install with: pip install scikit-learn"
            )

        # Load annotations
        texts: list[str] = []
        labels: list[int] = []
        with open(annotations_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "code_prefix" in entry and "score" in entry:
                        texts.append(entry["code_prefix"])
                        labels.append(int(entry["score"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        if len(texts) < 10:
            raise ValueError(f"Need at least 10 annotations, got {len(texts)}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_fraction, random_state=42,
        )

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Logistic regression
        model = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            class_weight="balanced",
        )
        model.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Save
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open(out / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save metadata
        meta = {
            "type": "tfidf_lr",
            "num_train": len(X_train),
            "num_test": len(X_test),
            "accuracy": round(acc, 4),
            "mae": round(mae, 4),
            "num_classes": len(set(labels)),
        }
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return ClassifierMetrics(
            accuracy=acc,
            mean_absolute_error=mae,
            num_train=len(X_train),
            num_test=len(X_test),
        )


class QualityClassifier:
    """Fast inference with a trained quality classifier."""

    def __init__(self, model_dir: str) -> None:
        path = Path(model_dir)
        with open(path / "vectorizer.pkl", "rb") as f:
            self._vectorizer = pickle.load(f)
        with open(path / "model.pkl", "rb") as f:
            self._model = pickle.load(f)

    def predict(self, code: str) -> float:
        """Predict quality score 0.0-1.0."""
        X = self._vectorizer.transform([code])
        pred = self._model.predict(X)[0]
        return float(pred) / 5.0

    def predict_batch(self, codes: list[str]) -> list[float]:
        """Batch prediction."""
        if not codes:
            return []
        X = self._vectorizer.transform(codes)
        preds = self._model.predict(X)
        return [float(p) / 5.0 for p in preds]


class ClassifierScorer:
    """Wraps QualityClassifier as a ScorerProtocol implementor."""

    name: str = "classifier"

    def __init__(self, model_dir: str = "models/quality_classifier") -> None:
        self._model_dir = model_dir
        self._classifier: QualityClassifier | None = None

    def _get_classifier(self) -> QualityClassifier:
        if self._classifier is None:
            self._classifier = QualityClassifier(self._model_dir)
        return self._classifier

    def score(self, code: str, metadata: dict[str, object] | None = None) -> ScorerResult:
        try:
            pred = self._get_classifier().predict(code)
            return ScorerResult(score=pred, scorer_name=self.name)
        except (FileNotFoundError, Exception) as e:
            return ScorerResult(
                score=0.5, scorer_name=self.name,
                details={"error": True, "message": str(e)},
            )

    def score_batch(self, items: list[tuple[str, dict[str, object] | None]]) -> list[ScorerResult]:
        codes = [code for code, _ in items]
        try:
            scores = self._get_classifier().predict_batch(codes)
            return [ScorerResult(score=s, scorer_name=self.name) for s in scores]
        except (FileNotFoundError, Exception) as e:
            return [
                ScorerResult(score=0.5, scorer_name=self.name, details={"error": True, "message": str(e)})
                for _ in items
            ]

    @staticmethod
    def is_available() -> bool:
        """Check if a trained model exists."""
        return Path("models/quality_classifier/model.pkl").exists()
