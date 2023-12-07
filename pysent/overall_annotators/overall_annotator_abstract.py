from abc import ABC, ABCMeta, abstractmethod
from pysent.data_structures import SentimentAnnotation


class OverallAnnotatorAbstract(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def classify(self, texts: list[str]) -> list[SentimentAnnotation]:
        """This method assign
        sentiment to texts.

        Parameters
        ----------
        texts : list[str]
            List of texts to analyse.

        Returns
        -------
        list[AspectAnnotation]
            List of annotated aspects.
        """
        return

    @staticmethod
    def check_arguments(texts: list[str]):
        """Checks if the arguments for aspects extraction are valid.

        Parameters
        ----------
        texts : list[str]
            List of texts.

        Raises
        ------
        ValueError
            Throw error if the argument is not a list.
        ValueError
            Throw error if at least one of the elements of the
            list is not a string.
        """
        if type(texts) != list:
            raise ValueError("You have to provide a list of strings!")

        bad_types = set([type(text) for text in texts if type(text) != str])
        if len(bad_types) != 0:
            raise ValueError(
                f"You have to provide a list of strings, not elements of {bad_types}!"
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
