"""
Abstract class that act as interface for classifiers. Classifiers are 
classes that assign sentiment to aspects and context.
"""

from abc import ABC, ABCMeta, abstractmethod
from itertools import chain
from pysent.data_structures import AspectAnnotation, ExtractedAspect


class AspectClassifier(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def classify(
        self, aspects: list[list[ExtractedAspect]], texts: list[str]
    ) -> list[AspectAnnotation]:
        """
        Extracts aspects and turn them into annotations.

        Parameters
        ----------
        aspects : list[list[ExtractedAspect]]
            List of lists containing extracted aspects. Each element
            of the list is a list of strings containing the keywords and a corresponding text.
        texts: list[str]
            List of original texts.

        Returns
        -------
        list[AspectAnnotation]
            List of annotated aspects.
        """
        return

    @staticmethod
    def check_arguments(aspects: list[list[ExtractedAspect]], texts):
        """Checks if the arguments for aspects extraction are valid.

        Parameters
        ----------
        aspects : list[list[ExtractedAspect]]
            List of list of aspects. One list per one text.

        Raises
        ------
        ValueError
            Throw error if the argument is not a list.
        ValueError
            Throw error if at least one of the elements of the
            list is not a string.
        """
        if type(aspects) != list | type(texts) != list:
            raise ValueError("You have to provide a list with aspects and texts!")

        if len(aspects) != len(texts):
            raise ValueError("Aspects must be of the same length as the texts!")

        bad_types = set([type(aspect) for aspect in aspects if type(aspect) != list])
        if len(bad_types) != 0:
            raise ValueError(
                f"You have to provide a list of lists, not elements of {bad_types}!"
            )

        next_types = set(
            [
                type(aspect)
                for aspect in list(chain.from_iterable(aspects))
                if type(aspect) != ExtractedAspect
            ]
        )
        if len(next_types) != 0:
            raise ValueError(
                f"You have to provide a list of lists with aspects, not elements of {bad_types}!"
            )

    @staticmethod
    def map_sentiment(sentiment: int) -> str:
        """Maps integer values of sentiment to strings

        Parameters
        ----------
        sentiment : int
            Sentiment as integer

        Returns
        -------
        str
            Sentiment as string label
        """
        if sentiment == 0:
            return "positive"
        if sentiment == 1:
            return "negative"
        if sentiment == 2:
            return "neutral"
