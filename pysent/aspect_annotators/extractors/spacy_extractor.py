"""
Sentiment classifier based on the spacy Python package.
"""

import spacy

from pysent.aspect_annotators.extractors.aspect_extractor import AspectExtractor
from pysent.data_structures import ExtractedAspect


class SpacyExtractor(AspectExtractor):
    def __init__(self, n_neighbors: int = 4, language="en"):
        """Object constructor

        Parameters
        ----------
        n_neighbors : int, optional
            Number of surroding words to be taken while extracting context,
            by default 4
        language : str, optional
            Language to use, one of ['pl', 'en'], by default "en"

        Raises
        ------
        ValueError
            Error is language not supported
        """
        if language not in ["en", "pl"]:
            raise ValueError("Language must be either 'en' or 'pl'!")
        self.n_neighbors = n_neighbors
        self.annotator = spacy.load(language + "_core_web_sm")

    def extract(self, texts: list[str]) -> list[list[ExtractedAspect]]:
        super().check_arguments(texts)
        aspects = []

        for text in texts:
            doc = self.annotator(text)
            sentence = next(doc.sents)
            subjects = [
                word.orth_
                for word in sentence
                if word.dep_ in ["nsubj"] and word.orth_ not in ["I", "you"]
            ]

            extracted_aspect = [
                ExtractedAspect(
                    aspect=aspect,
                    text=self.get_neighbors(
                        main_word=aspect, full_text=text, n_neighbors=self.n_neighbors
                    ),
                )
                for aspect in subjects
            ]
            aspects.append(extracted_aspect)

        return aspects
