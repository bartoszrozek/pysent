"""
Sentiment extrassifier based on the ChatGPT.
"""
from openai import OpenAI
import openai
import time

from pysent.aspect_annotators.extrassifiers.aspect_extrassifier import (
    AspectExtrassifier,
)
from pysent.data_structures import (
    AspectAnnotation,
    ExtractedAspect,
    SentimentAnnotation,
)


class ChatGPTExtrassifier(AspectExtrassifier):
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

    def classify(self, texts: list[str]) -> list[AspectAnnotation]:
        super().check_arguments(texts)

        annotations = []
        for text in texts:
            message = [
                {
                    "role": "system",
                    "content": f"""For text below provide me an aspect based sentiment analysis and score in the format:
                    Aspect: <aspect you suggest, exact words from text, do not change its form>
                    Label: <label you suggest>
                    Score: <score you suggest>

                    Don't repeat the text or provide any additional output. Answer should be in the same language as the input.

                    Text:
                    {text}
                    """,
                }
            ]
            chat = OpenAI(api_key=openai.api_key).chat
            chat_completion = chat.completions.create(
                messages=message,
                model="gpt-3.5-turbo",
            )
            reply = chat_completion.choices[0].message.content
            # wait to fit into OpenAI 3RequestsPerMinute restriction
            if self.free_tier:
                time.sleep(20)
            try:
                aspects_list = []
                split_by_aspects = reply.split("\n\n")
                for aspect_label_score in split_by_aspects:
                    aspect, label, score = aspect_label_score.split("\n")
                    aspect = aspect.removeprefix("Aspect:")
                    label = label.removeprefix("Label: ")
                    score = score.removeprefix("Score: ")
                    score = float(score)
                    aspects_list.append(
                        SentimentAnnotation(text=aspect, label=label, score=score)
                    )
            except:
                ValueError(f"Something wrong in the response: {reply}!")

            annotations.append(AspectAnnotation(text=text, aspects=aspects_list))

        return annotations
