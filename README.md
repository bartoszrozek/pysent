# pysent

Python package with wrappers for different methods of sentiment and sentiment based analysis.

## Installation

Pip:

```sh
pip install git+https://github.com/bartoszrozek/pysent
```

## Examples

Example use (overall annotation):

```python
>>> from pysent import FlairAnnotator, OverallAnotator
>>> flair_an = FlairAnnotator()
>>> annotator = OverallAnotator(flair_an)
>>> annotation = annotator.annotate("This book is really nice!")
>>> print(annotation)

... [SentimentAnnotation(text='This book is really nice!', label='positive', score=0.9575109481811523)]
```

Example use (aspect based annotation):

```python
>>> from pysent import AspectAnotator
>>> import pysent.aspect_annotators.extractors as extractors
>>> import pysent.aspect_annotators.classifiers as classifiers
>>> import pysent.aspect_annotators.extrassifiers as extrassifiers

>>> spacy_ext = extractors.SpacyExtractor()
>>> flair_cl = classifiers.FlairClassifier()
>>> annotator = AspectAnotator([spacy_ext, flair_cl])
>>> result = annotator.annotate("This book is really nice!")

... [AspectAnnotation(text='This book is really nice!', aspects=[SentimentAnnotation(text='book', label='positive', score=0.9575109481811523)])]
```
