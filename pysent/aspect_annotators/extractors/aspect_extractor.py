"""
Abstract class that act as interface for extractors. Extractors are 
classes that extracts aspects and context from texts.
"""

from abc import ABC, ABCMeta, abstractmethod
from pysent.data_structures import AspectAnnotation, ExtractedAspect


class AspectExtractor(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def extract(self, texts: list[str]) -> list[list[ExtractedAspect]]:
        """This method extracts aspects keywords from the given texts.

        Parameters
        ----------
        texts : list[str]
            List of strings to extract aspects keywords from.

        Returns
        -------
        list[list[str]]
            List of aspects keywords extracted from the given texts. Each element
            of the list is a list of strings containing the keywords and a corresponding text.
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
    def get_neighbors(main_word, full_text: str, n_neighbors: int = 3) -> str:
        """Taking out the context for the aspect, by including words surrounding
        the aspect.

        Parameters
        ----------
        main_word : str
            Word for which we extract the context.
        text : str
            Text from which the aspect was taken.
        n_neighbors : int
            Number of words in the sentence to be taken next to the aspect, by default 3

        Returns
        -------
        str
            Aspect with the context (close words).
        """
        list_of_words = full_text.replace("'", " ").split()
        try:
            word_position = list_of_words.index(main_word)
            start_idx = max(0, word_position - n_neighbors)
            stop_idx = min(len(list_of_words), word_position + n_neighbors + 1)
            context = " ".join(list_of_words[start_idx:stop_idx])
        except:
            context = full_text
        return context
