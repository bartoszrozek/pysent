.. pysent documentation master file, created by
   sphinx-quickstart on Wed Dec  6 20:37:04 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pysent's documentation!
==================================

OverallAnnotator
==================================
OverallAnnotator is a class that as an input has a tool class that inherits from OverallAnnotatorAbstract (described further). It has methods that allow to generate sentiment annotation for text and test the supplied annotator with given gold standard annotations. Its structure is a bit too complicated since all of the methods could be implemented in the OverallAnnotatorAbstract class, but we decided to follow the path of AspectAnnotator and to keep it easily extensible.

Overall annotators are classed that inherit from OverallAnnotatorAbstract class. The OverallAnnotatorAbstract class is an interface that has methods check\_arguments and classify. We implemented three tools that can assign sentiment to the given text:

* FlairAnnotator - a class based on the flair Python package
* SentiAnnotator - a class based on the SentiStrength tool and sentistrength Python package which is a CLI wrapper for SentiStrength
* ChatGPTAnnotator - a class based on the OpenAI Python package that allows querying ChatGPT from Python and prompt engineering done by us.


Extractors
==================================
Extractors are classes that inherit from the interface AspectExtractor and implement method extract which extracts aspects from sentences with context. In the package, there are three extractors implemented: 

* SpacyExtractor - a class based on the Python package spacy, takes out aspects by part of speech in the sentence, the context is taken out as a set number of words surrounding the aspect.
* PyabsaExtractor - a class based on the Python package pyabsa, the context is taken out as a set number of words surrounding the aspect.  
* ChatGPTExtractor - a class based on the Python package OpenAI and prompt engineering. 

Classifiers
==================================
Classifiers are classes that inherit from the interface AspectClassifier and implement method classify which for the given context and aspect returns sentiment label. In the package, there are three classifiers implemented: 

* FlairClassifier - a class based on the Python package flair, assigns sentiment to the given context, the aspect is taken from the extractor. 
* SentiClassifier - a class based on the tool Sentistrenght and Python package sentistrength, the aspects is taken from the extractor, but the context is extracted by Sentistrength itself.

Originally, sentistrength Python wrapper does not provide the user with the option to do an aspect-based analysis which is possible in the SentiStrength tools itself. To cover this, we have decided to add this functionality and contribute to this open-source package. Our pull request is currently opened - \href{https://github.com/zhunhung/Python-SentiStrength/pull/16{github link.

Extrassifiers 
==================================
Extrassifiers are classes that inherit from the interface AspectExtrassifier and implement method classify that extracts aspects and assigns sentiment to them. The name is taken as a combination of two task names - "extract" and "classify". In the package, there are two extrassifiers implemented: 

* PyabsaExtrassifier - a class based on the Python package pyabsa.  
* ChatGPTExtractor - a class based on the Python package OpenAI and prompt engineering. 

Data structures
==================================

To keep the appropriate structure, we decided to introduce several data classes placed in the file \textit{data\_structures.py:
* ExtractedAspect - class representing the output of extraction aspect tools. Has an aspect keyword and context in text.
* SentimentAnnotation - Contains information about single sentiment annotation.
* AspectAnnotation - Contains information about aspect sentiment annotation.
* OrdinaryResults - Contains results for the overall annotation where only the label is predicted.
* AspectBasedResults - Contains results for the aspect-based annotation where the model predicts the place of the annotation and the label.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   annotators
   overall_annotators
   extractors
   classifiers
   extrassifiers
   data_structures


   
   