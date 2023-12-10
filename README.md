# pysent

Python package with wrappers for different methods of sentiment and sentiment based analysis.

## Documentation

The full documentation is available at [GCP bucket](https://storage.googleapis.com/nlp_bucket420/html/index.html?fbclid=IwAR0AyYC11tef7D4MWJXs5mKOetqCiekaooKSrjSgS2Egslzwie6L3SsRb9w)

## Installation

Pip:

```sh
pip install git+https://github.com/bartoszrozek/pysent
```

## Examples

Example use (overall annotation):

```python
>>> from pysent import OverallAnotator
>>> from pysent.overall_annotators import FlairAnnotator
>>> flair_an = FlairAnnotator()
>>> annotator = OverallAnotator(flair_an)
>>> annotation = annotator.annotate("This book is really nice!")
>>> print(annotation)

... [SentimentAnnotation(text='This book is really nice!',
...  label='positive', score=0.9575109481811523)]
```

Example use (aspect based annotation):

```python
>>> from pysent import AspectAnotator
>>> import pysent.aspect_annotators.extractors as extractors
>>> import pysent.aspect_annotators.classifiers as classifiers

>>> spacy_ext = extractors.SpacyExtractor()
>>> flair_cl = classifiers.FlairClassifier()
>>> annotator = AspectAnotator([spacy_ext, flair_cl])
>>> result = annotator.annotate("This book is really nice!")

... [AspectAnnotation(text='This book is really nice!',
..   aspects=[SentimentAnnotation(text='book', label='positive', score=0.9575109481811523)])]
```
