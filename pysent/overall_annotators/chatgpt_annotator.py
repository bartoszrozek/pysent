"""
Sentiment annotator based on the Chat GPT.
"""
from openai import OpenAI
import openai
import time

from pysent.overall_annotators.overall_annotator_abstract import (
    OverallAnnotatorAbstract,
)
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class ChatGPTAnnotator(OverallAnnotatorAbstract):
    def __init__(self, api_key: str, free_tier: bool = True):
        """Object constructor

        Parameters
        ----------
        api_key : str
            API key to the Open AI service
        free_tier : bool, optional
            Indicator whether account connected with the API key is free.
            For free account, the cooldown is added, by default True
        """
        openai.api_key = api_key
        self.free_tier = free_tier

    def classify(self, texts: str) -> list[SentimentAnnotation]:
        super().check_arguments(texts)

        annotations = []

        for text in texts:
            if len(text) > 1000:
                text = text[:1000]
            message = [
                {
                    "role": "system",
                    "content": f"""For text below provide me a sentiment analysis label and score in the format:
                                    Label: <label you suggest>
                                    Score: <score you suggest>
                                    
                                    Text:
                                    {text}""",
                }
            ]
            chat = OpenAI(api_key=openai.api_key).chat
            chat_completion = chat.completions.create(
                messages=message,
                model="gpt-3.5-turbo",
            )
            reply = chat_completion.choices[0].message.content

            try:
                label, score = reply.split("\n")
                label = label.removeprefix("Label: ")
                score = score.removeprefix("Score: ")
                score = float(score)
            except:
                ValueError(f"Something wrong in the response: {reply}!")
            if self.free_tier:
                time.sleep(20)

            annotations.append(SentimentAnnotation(text=text, label=label, score=score))
        return annotations
