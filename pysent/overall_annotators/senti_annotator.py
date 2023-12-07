"""
Sentiment annotator based on the Flair Python package.
"""

from sentistrength import PySentiStr
import numpy as np
from itertools import chain
from pysent.overall_annotators.overall_annotator_abstract import (
    OverallAnnotatorAbstract,
)
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class SentiAnnotator(OverallAnnotatorAbstract):
    def __init__(
        self, language: str = "en", ss_jar_path: str = None, ss_lang_path: str = None
    ):
        """Object constructor

        Parameters
        ----------
        language : str, optional
            Language to use, one of ['pl', 'en'], by default "en"
        ss_jar_path : str, optional
            Path to jar SentiStrenght file, by default None
        ss_lang_path : str, optional
            Path to jar SentiStrenght language folder, by default None

        Raises
        ------
        ValueError
            Error if .jar path is missing.
        ValueError
            Error if data folder path is missing.
        ValueError
            Error if language is not one of ['en', 'pl'].
        """
        if ss_jar_path == None:
            raise ValueError(
                "Missing path for SentiStrength .jar! Note: Provide absolute path instead of relative path"
            )
        if ss_jar_path == None:
            raise ValueError(
                "Missing path for SentiStrength data folder! Note: Provide absolute path instead of relative path"
            )
        if language not in ["en", "pl"]:
            raise ValueError("Language must be either 'en' or 'pl'!")
        self.classifier = PySentiStr()
        self.classifier.setSentiStrengthPath(ss_jar_path)
        self.classifier.setSentiStrengthLanguageFolderPath(ss_lang_path)

    def classify(self, texts: str) -> list[SentimentAnnotation]:
        super().check_arguments(texts)

        annotations = []

        for text in texts:
            sentiment_score = self.classifier.getSentiment(text, score="trinary")
            sentiment_index = [
                np.argmax(np.abs(sentiment)) for sentiment in sentiment_score
            ]
            sentiment = list(map(self.map_sentiment, sentiment_index))[0]
            annotations.append(
                SentimentAnnotation(
                    text=text,
                    label=sentiment,
                    score=sentiment_score[0][sentiment_index[0]],
                )
            )
        return annotations
