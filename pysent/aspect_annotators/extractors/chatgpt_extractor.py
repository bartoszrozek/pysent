"""
Sentiment extractor based on the ChatGPT.
"""
from openai import OpenAI
import openai
import time

from pysent.aspect_annotators.extractors.aspect_extractor import AspectExtractor
from pysent.data_structures import ExtractedAspect


class ChatGPTExtractor(AspectExtractor):
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

    def extract(self, texts: list[str]) -> list[list[ExtractedAspect]]:
        super().check_arguments(texts)
        aspects = []

        for text in texts:
            text_aspects = []
            message = [
                {
                    "role": "system",
                    "content": f"""For Text below provide me it's distinct aspects - subjects present in the text, that can be later used for aspect based sentiment analysis. 
                You can return one or multiple aspects, but they shouldn't repeat. 
                If there are two chunks about the same thing in the text, you should find a way to distinguish them in the aspect name.
                Keep the aspects concise (no more than 3 words). 
                Besides the aspect, provide the chunk of the text that describes that aspect. Chunks should sum up to the entire sentence.
                If you are returning one aspect, use the format:
                Aspect: <aspect you suggest, exact words from text, do not change its form>
                Chunk: <chunk of the Text about the aspect>

                If you are returning multiple aspects, use the format:
                Aspect: <aspect you suggest, exact words from text, do not change its form>
                Chunk: <piece of the Text about the aspect>

                Aspect: <aspect you suggest, exact words from text, do not change its form>
                Chunk: <piece of the Text about the aspect>
                ... and so on.
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
            try:
                aspects_list = []
                split_by_aspects = reply.split("\n\n")
                for aspect_chunk in split_by_aspects:
                    aspect, chunk = aspect_chunk.split("\n")
                    aspect = aspect.removeprefix("Aspect: ")
                    chunk = chunk.removeprefix("Chunk: ")
                    text_aspects.append(ExtractedAspect(aspect, chunk))

                if self.free_tier:
                    time.sleep(20)
            except:
                ValueError(f"Something wrong in the response: {reply}!")

            aspects.append(text_aspects)
        return aspects
