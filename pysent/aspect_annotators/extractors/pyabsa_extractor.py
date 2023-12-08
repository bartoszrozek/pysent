"""
Sentiment classifier based on the PyABSA Python package.
"""
from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints


from pysent.aspect_annotators.extractors.aspect_extractor import AspectExtractor
from pysent.data_structures import ExtractedAspect


class PyabsaExtractor(AspectExtractor):
    def __init__(self, n_neighbors: int = 4):
        """Object constructor

        Parameters
        ----------
        n_neighbors : int, optional
            Number of surroding words to be taken while extracting context,
            by default 4

        Raises
        ------
        ValueError
            Error is language not supported
        """
        checkpoint_map = available_checkpoints()

        self.classifier = ATEPC.AspectExtractor(
            "multilingual",
            auto_device=False,  # True,  # False means load model on CPU
            cal_perplexity=True,
        )
        self.n_neighbors = n_neighbors

    def extract(self, texts: list[str]) -> list[list[ExtractedAspect]]:
        super().check_arguments(texts)

        tool_annotations = self.classifier.predict(
            texts,
            save_result=False,
            print_result=False,  # print the result
            ignore_error=True,  # ignore the error when the model cannot predict the input
        )
        aspects = []

        for anotation in tool_annotations:
            aspects.append(
                [
                    ExtractedAspect(
                        aspect=aspect,
                        text=self.get_neighbors(
                            main_word=aspect,
                            full_text=anotation["sentence"],
                            n_neighbors=self.n_neighbors,
                        ),
                    )
                    for aspect in anotation["aspect"]
                ]
            )

        return aspects
