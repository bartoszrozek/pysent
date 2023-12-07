"""
Sentiment annotator based on the Flair Python package.
"""

from flair.data import Sentence
from flair.nn import Classifier
from itertools import chain

from pysent.overall_annotators.overall_annotator_abstract import (
    OverallAnnotatorAbstract,
)
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class FlairAnnotator(OverallAnnotatorAbstract):
    def __init__(self, language: str = "en"):
        """Object constructor

        Parameters
        ----------
        language : str, optional
            Language to use, one of ['pl', 'en'], by default "en"

        Raises
        ------
        ValueError
            Error is language not supported
        """
        if language not in ["en", "pl"]:
            raise ValueError("Language must be either 'en' or 'pl'!")
        self.classifier = Classifier.load("sentiment")

    def classify(self, texts: str) -> list[SentimentAnnotation]:
        super().check_arguments(texts)

        annotations = []

        for text in texts:
            sentence = Sentence(text)
            self.classifier.predict(sentence)
            annotations.append(
                SentimentAnnotation(
                    text=text, label=sentence.tag.lower(), score=sentence.score
                )
            )
        return annotations
