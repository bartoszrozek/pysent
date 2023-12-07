"""
Class that wraps up the annotators. The class also allows user to perform
sentiment analysis and test tools on already annotated texts. The structure is 
similar to the aspect based class.
"""

from typing import Literal
from pysent.data_structures import SentimentAnnotation, OrdinaryResults
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from pysent.overall_annotators import OverallAnnotatorAbstract


class OverallAnotator:
    def __init__(self, tool: OverallAnnotatorAbstract) -> None:
        """Wrapper for the overall annotators classes.

        Parameters
        ----------
        tool : OverallAnnotatorAbstract
            Tool which performs the sentiment analysis.

        Raises
        ------
        ValueError
            If elements of pipeline do not meet the criteria above.
        """

        if not isinstance(tool, OverallAnnotatorAbstract):
            raise ValueError("Tool must be (inherit from) an AspectExtrassifier class!")

        self.tool = tool

    def annotate(self, texts: list[str]) -> list[SentimentAnnotation]:
        """Extracts and annotates aspects from the given texts.

        Parameters
        ----------
        texts : list[str]
            List of texts to annotate.

        Returns
        -------
        list[SentimentAnnotation]
            List of sentiment annotations, the same length as the given texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        annotations = self.tool.classify(texts)

        return annotations

    def test_annotator(
        self, texts: list[str], true_labels: list[str]
    ) -> OrdinaryResults:
        predicted_labels = self.annotate(texts)
        predicted_labels = [
            predicted_label.label.lower() for predicted_label in predicted_labels
        ]
        true_labels = [true_label.lower() for true_label in true_labels]
        results = self.calculate_results(predicted_labels, true_labels)
        return results

    def calculate_results(self, true_labels, predicted_labels) -> OrdinaryResults:
        if len(true_labels) != len(predicted_labels):
            raise ValueError(
                "Lenghts of true_labels and predicted_labels must be equal!"
            )

        return OrdinaryResults(
            global_accuracy=accuracy_score(y_true=true_labels, y_pred=predicted_labels),
            macro_precision=precision_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division=0,
            ),
            macro_recall=recall_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division=0,
            ),
            macro_f1=f1_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="macro",
                zero_division=0,
            ),
            micro_precision=precision_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division=0,
            ),
            micro_recall=recall_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division=0,
            ),
            micro_f1=f1_score(
                y_true=true_labels,
                y_pred=predicted_labels,
                average="micro",
                zero_division=0,
            ),
            name=type(self.tool).__name__,
        )
