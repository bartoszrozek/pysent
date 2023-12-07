"""
Class that connects aspect extractors and aspect classifiers and 
wraps them into one. The class also allows user to perform aspect based 
sentiment analysis and test tools on already annotated texts.
"""

from typing import Literal
from pysent.data_structures import (
    SentimentAnnotation,
    AspectAnnotation,
    AspectBasedResults,
)
from pysent.transforms import transform_aspects
import numpy as np
import pandas as pd

from pysent.aspect_annotators.extractors import AspectExtractor
from pysent.aspect_annotators.classifiers import AspectClassifier
from pysent.aspect_annotators.extrassifiers import AspectExtrassifier


class AspectAnotator:
    def __init__(self, pipeline: list) -> None:
        """Connector for aspect extractors and aspect classifiers or wrapper for
        classes that incorporates both of them.

        Parameters
        ----------
        pipeline : list
            List containing elements of the pipeline of aspect based sentiment analysis.
            Two options are available:
                - one element list - contains object that can do the whole process (inherits
                from AspectExtrassifier class)
                - two elements list - one object extracts aspects (inherits from AspectExtractor)
                and second object classify aspects (inherits from AspectClassifier)

        Raises
        ------
        ValueError
            If elements of pipeline do not meet the criteria above.
        """
        if len(pipeline) > 2:
            raise ValueError("Pipeline longer than 2 elements is not allowed!")

        if len(pipeline) == 2:
            if not isinstance(pipeline[0], AspectExtractor):
                raise ValueError(
                    "First element of the pipeline list must be (inherit from) an AspectExtractor class!"
                )

            if not isinstance(pipeline[1], AspectClassifier):
                raise ValueError(
                    "Second element of the pipeline list must be (inherit from) an AspectClassifier class!"
                )

        if len(pipeline) == 0:
            if not isinstance(pipeline[0], AspectExtrassifier):
                raise ValueError(
                    "Only element of the pipeline list must be (inherit from) an AspectExtrassifier class!"
                )

        self.pipeline = pipeline

    def annotate(self, texts: list[str]) -> list[AspectAnnotation]:
        """Extracts and annotates aspects from the given texts.

        Parameters
        ----------
        texts : list[str]
            List of texts to annotate.

        Returns
        -------
        list[AspectAnnotation]
            List of aspects with sentiment, the same length as the given texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(self.pipeline) == 2:
            extractor = self.pipeline[0]
            classifier = self.pipeline[1]

            aspects = extractor.extract(texts)
            annotations = classifier.classify(aspects, texts)

        if len(self.pipeline) == 1:
            extrassifier = self.pipeline[0]
            annotations = extrassifier.classify(texts)

        return annotations

    def test_annotator(
        self,
        true_annotations: list[AspectAnnotation] | pd.DataFrame,
        id_column: str = None,
        text_column: str = None,
        aspect_column: str = None,
        sentiment_column: str = None,
    ) -> AspectBasedResults:
        """Tests provided tools in context of aspect based analysis.

        Parameters
        ----------
        true_annotations : list[AspectAnnotation] | pd.DataFrame
            Annotated text in the form of list of AspectAnnotations or Data Frame where each row contains
            aspect,full text and label.
        id_column : str, optional
            If true_annotations is a pandas data frame, name of the column with id, by default None
        text_column : str, optional
            If true_annotations is a pandas data frame, name of the column with full text, by default None
        aspect_column : str, optional
            If true_annotations is a pandas data frame, name of the column with extracted aspect, by default None
        sentiment_column : str, optional
            If true_annotations is a pandas data frame, name of the column with assigned sentiment, by default None

        Returns
        -------
        AspectBasedResults
            Object with results, class AspectBasedResults

        Raises
        ------
        ValueError
            If columns are missing
        """
        if isinstance(true_annotations, pd.DataFrame):
            for col in [id_column, text_column, aspect_column, sentiment_column]:
                if col is None:
                    raise ValueError(f"Specify {col} if data frame is provided!")
            true_annotations = transform_aspects(
                true_annotations,
                id_column,
                text_column,
                aspect_column,
                sentiment_column,
            )
        texts = [aa.text for aa in true_annotations]
        predicted_annotations = self.annotate(texts)
        results = self.calculate_results(true_annotations, predicted_annotations)
        return results

    def calculate_results(
        self,
        true_annotations: list[AspectAnnotation],
        predicted_annotations: list[AspectAnnotation],
    ) -> AspectBasedResults:
        """Calculates the results based on missmatches between true annotations and predictions.

        Parameters
        ----------
        true_annotations : list[AspectAnnotation]
            List of true annotations
        predicted_annotations : list[AspectAnnotation]
            List of predicted annotations

        Returns
        -------
        AspectBasedResults
            Object with results, class AspectBasedResults
        """
        COR = 0
        INC = 0
        PAR = 0
        SPU = 0

        if isinstance(true_annotations, pd.DataFrame):
            true_annotations = transform_aspects(true_annotations)

        for pred_an, true_an in zip(predicted_annotations, true_annotations):
            for pred in pred_an.aspects:
                pred_aspect = pred.text
                pred_sentiment = pred.label.lower()
                matching_aspect = False
                for true_ in true_an.aspects:
                    true_aspect = true_.text
                    true_sentiment = true_.label.lower()
                    if pred_aspect == true_aspect:
                        if pred_sentiment == true_sentiment:
                            COR += 1
                        else:
                            INC += 1
                        matching_aspect = True
                        break
                if not matching_aspect:
                    for true_ in true_an.aspects:
                        true_aspect = true_.text
                        true_sentiment = true_.label.lower()
                        if true_aspect in pred_aspect or pred_aspect in true_aspect:
                            if pred_sentiment == true_sentiment:
                                PAR += 1
                            else:
                                INC += 1
                            matching_aspect = True
                            break
                if not matching_aspect:
                    SPU += 1

        MIS = np.sum([len(ta.aspects) for ta in true_annotations]) - COR - INC - PAR

        name = " + ".join([type(tool).__name__ for tool in self.pipeline])
        return AspectBasedResults(
            correct=COR,
            incorrect=INC,
            missing=MIS,
            partial=PAR,
            spurious=SPU,
            name=name,
        )
