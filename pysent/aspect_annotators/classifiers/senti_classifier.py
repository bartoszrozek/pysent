"""
Sentiment classifier based on the SentiStrenght tool and Python wrapper for it - PySentiStr package.
"""

from sentistrength import PySentiStr
import numpy as np
from itertools import chain

from pysent.aspect_annotators.classifiers.aspect_classifer import AspectClassifier
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class SentiClassifier(AspectClassifier):
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

    def classify(
        self, aspects: list[list[ExtractedAspect]], texts: str
    ) -> list[AspectAnnotation]:
        super().check_arguments(aspects, texts)

        rep = [len(a) for a in aspects]
        texts_rep = np.repeat(texts, rep)
        aspects_unlist = [el.aspect for el in list(chain.from_iterable((aspects)))]

        sentiments = self.classifier.getSentiment(
            texts_rep, keywords=aspects_unlist, score="trinary"
        )
        sentiments = [np.argmax(np.abs(sentiment)) for sentiment in sentiments]
        sentiments = list(map(self.map_sentiment, sentiments))
        sentiments = [
            SentimentAnnotation(text=aspects_unlist[i], label=sentiments[i])
            for i in range(len(sentiments))
        ]

        rep_cum = np.insert(np.cumsum(rep), 0, 0, axis=0)
        sentiments_grouped = [
            sentiments[rep_cum[i] : rep_cum[i + 1]] for i in range(len(rep_cum) - 1)
        ]
        annotations = [
            AspectAnnotation(text=texts[i], aspects=sentiments_grouped[i])
            for i in range(len(sentiments_grouped))
        ]

        return annotations
