"""
Abstract class that act as interface for extrassifier. Extrassifier are 
classes that do both - first extracts aspects and then assign aspects to them.
"""

from abc import ABC, ABCMeta, abstractmethod
from pysent.data_structures import AspectAnnotation


class AspectExtrassifier(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def classify(self, texts: list[str]) -> list[AspectAnnotation]:
        """This method extracts aspects keywords from the given texts and assign
        sentiment to them.

        Parameters
        ----------
        texts : list[str]
            List of strings to analyse.

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
