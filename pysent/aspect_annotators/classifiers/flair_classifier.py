"""
Sentiment classifier based on the Flair Python package.
"""

from flair.data import Sentence
from flair.nn import Classifier
from itertools import chain

from pysent.aspect_annotators.classifiers.aspect_classifer import AspectClassifier
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class FlairClassifier(AspectClassifier):
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

    def classify(
        self, aspects: list[list[ExtractedAspect]], texts: str
    ) -> list[AspectAnnotation]:
        super().check_arguments(aspects, texts)

        annotations = []

        for text_aspects, text in zip(aspects, texts):
            aspects_list = []
            for extracted_aspect in text_aspects:
                aspect = extracted_aspect.aspect
                chunk = extracted_aspect.text
                sentence = Sentence(chunk)
                self.classifier.predict(sentence)
                aspects_list.append(
                    SentimentAnnotation(
                        text=aspect, label=sentence.tag.lower(), score=sentence.score
                    )
                )
            annotations.append(AspectAnnotation(text=text, aspects=aspects_list))

        return annotations
