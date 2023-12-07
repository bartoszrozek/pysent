"""
Sentiment extrassifier based on the pyabsa Python package.
"""

from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints

from pysent.aspect_annotators.extrassifiers.aspect_extrassifier import (
    AspectExtrassifier,
)
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class PyabsaExtrassifier(AspectExtrassifier):
    def __init__(self):
        checkpoint_map = available_checkpoints()

        self.classifier = ATEPC.AspectExtractor(
            "multilingual",
            auto_device=False,  # True,  # False means load model on CPU
            cal_perplexity=True,
        )

    def classify(self, texts: list[str]) -> list[AspectAnnotation]:
        super().check_arguments(texts)

        tool_annotations = self.classifier.predict(
            texts,
            save_result=False,
            print_result=False,  # print the result
            ignore_error=True,  # ignore the error when the model cannot predict the input
        )

        annotations = [
            AspectAnnotation(
                text=result["sentence"],
                aspects=[
                    SentimentAnnotation(text=aspect, label=sentiment, score=confidence)
                    for aspect, sentiment, confidence in zip(
                        result["aspect"],
                        result["sentiment"],
                        result["confidence"],
                    )
                ],
            )
            for result in tool_annotations
        ]
        return annotations
