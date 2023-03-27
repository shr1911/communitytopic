# Welcome to Community Topic’s documentation!

[![PyPI - Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://pypi.org/project/communitytopic/)
[![](https://img.shields.io/pypi/v/communitytopic.svg)](https://pypi.org/project/communitytopic/)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1npBdhbDI7c3NOfjhgbLYvUba6bLjAyUu?usp=sharing)
[![](https://readthedocs.org/projects/assets/badge/?version=latest)](https://shr1911.github.io/communitytopic/)
[![](https://img.shields.io/pypi/l/communitytopic.svg)](https://github.com/shr1911/communitytopic/blob/main/LICENSE)

## Introduction
- **What is Community Topic**?

In this repository we present our novel method called Community Topic for Topic Modelleling as a Pypi library. Our algorithm, Community Topic, is based on mining communities of terms from term-occurrence networks extracted from the documents. In addition to providing interpretable collections of terms as topics, the network representation provides a natural topic structure. The topics form a network, so topic similarity is inferred from the weights of the edges between them. Super-topics can be found by iteratively applying community detection on the topic network, grouping similar topics together. Sub-topics can be found by iteratively applying community detection on a single topic community. This can be done dynamically, with the user or conversation agent moving up and down the topic hierarchy as desired.

- **What problem does it solve? & Who is it for?**

Unfortunately, the most popular topic models in use today do not provide a suitable topic structure for these purposes and the state-of-the-art models based on neural networks suffer from many of the same drawbacks while requiring specialized hardware and many hours to train. This makes Community Topic an ideal topic modelling algorithm for both applied research and practical applications like conversational agents.
#

## Content

- Installation Guide
- Getting Started
- API usage
