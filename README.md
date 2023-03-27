[![PyPI - Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://pypi.org/project/communitytopic/)
[![](https://img.shields.io/pypi/v/communitytopic.svg)](https://pypi.org/project/communitytopic/)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1npBdhbDI7c3NOfjhgbLYvUba6bLjAyUu?usp=sharing)
[![](https://readthedocs.org/projects/assets/badge/?version=latest)](https://shr1911.github.io/communitytopic/)
[![](https://img.shields.io/pypi/l/communitytopic.svg)](https://github.com/shr1911/communitytopic/blob/main/LICENSE)

# Community Topic
## Introduction
- **What is Community Topic**?

In this repository we present our novel method called Community Topic for Topic Modelleling as a Pypi library. Our algorithm, Community Topic, is based on mining communities of terms from term-occurrence networks extracted from the documents. In addition to providing interpretable collections of terms as topics, the network representation provides a natural topic structure. The topics form a network, so topic similarity is inferred from the weights of the edges between them. Super-topics can be found by iteratively applying community detection on the topic network, grouping similar topics together. Sub-topics can be found by iteratively applying community detection on a single topic community. This can be done dynamically, with the user or conversation agent moving up and down the topic hierarchy as desired.

- **What problem does it solve? & Who is it for?**

Unfortunately, the most popular topic models in use today do not provide a suitable topic structure for these purposes and the state-of-the-art models based on neural networks suffer from many of the same drawbacks while requiring specialized hardware and many hours to train. This makes Community Topic an ideal topic modelling algorithm for both applied research and practical applications like conversational agents.


[**Website Documentation is here**](https://shr1911.github.io/communitytopic/)

## Requirement & Installation

- System requirement

      Python >= 3.6
      commodity hardware
      setuptools~=67.6.0
      spacy~=3.5.0
      numpy~=1.21.5
      gensim~=4.2.0
      networkx~=2.8.4
      igraph~=0.10.4


- Installation Option

  The easy way to install CommunityTopic is:

      pip install communitytopic
   
   
- Spacy models for different languages for pre-processing (Commuity Topic has a pre-processing function as shown in below getting started example, and it requires spacy model to be dowloaded via python for the language which we are using). Following are commands for the same :

| Language | Commond to download spacy model |
|---|---|
| English | !python -m spacy download en_core_web_sm |
| Italian | !python -m spacy download it_core_news_sm |
| French | !python -m spacy download fr_core_news_sm |
| German | !python -m spacy download de_core_news_sm |
| Spanish | !python -m spacy download es_core_news_sm |

      
## Datasets and Evaluation Metrics Used
We have used following **dataset for our experiment**.

| Name of the Dataset | Source  | Source Language |
|---|---|---|
| BBC | [BBC](https://www.kaggle.com/competitions/learn-ai-bbc/data) | English |
| 20Newsgroups | [20Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) | English |
| Reuters21578 | [Reuters21578](https://huggingface.co/datasets/reuters21578) | English |
| Europarl | [Europarl](https://www.statmt.org/europarl/) | English, Italian, French, German, Spanish |


Also we have used following **metrics for our Evaluation**:

**1. Coherences**
To compare different topic models, we use two coherence measures: c_v and c_npmi. Both measures have been shown to correlate with human judgements of topic quality with CV having the strongest correlation

**2. Diversity Measures**
- Proportion of unique words (PWU): Computes the proportion of unique words for the topic
- Average Pairwise Jaccard Diversity (PJD): Coomputes the average pairwise jaccard distance between the topics.
- Inverted Rank-Biased Overlap (IRBO): Computes score of the rank biased overlap over the topics. 

**3. Hierarchical Measures**
- Topic Specialization:  measures the distance of a topic’s probability distribution over terms from thegeneral probability distribution of all terms in the corpus given by their occurrence frequency. We expect topics at higher levels in the hierarchy closer to theroot to be more general and less specialized and topics further down the hierarchy to be more specialized
- Topic Affinity: measures the similarity between a super-topic and a set of sub-topics. We expect higher affinity between a parent topic and its children and lower affinity between a parent topic and sub-topics which are not its children

## Getting Started (Try it out)
This is an example tuotrial which finds topic of BBC dataset using best combination for Pre-Processing and Community Topic Algorithm. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1npBdhbDI7c3NOfjhgbLYvUba6bLjAyUu?usp=sharing)

> Step 1: import necessary class of the library
```python

from communitytopic import CommunityTopic
from communitytopic import PreProcessing
```

> Step 2: Load raw corpus as the dataset, here we are using BBC dataset. 
```python

with open("<Path-To-Train-Dataset>/bbc_train.txt", "r", encoding='utf-8') as f:
      bbc_train = f.read()
      
with open("<Path-To-Test-Dataset>/bbc_test.txt", "r", encoding='utf-8') as f:
      bbc_test = f.read()
```

> Step 3: Performing pre-processing on training and testing corpus
```python

tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
        train=bbc_train,
        test=bbc_test,
        ner=1,
        pos_filter=3,
        phrases="npmi",
        phrase_threshold=0.35,
        language="en")
```


> Step 4: Applying Community Topic algorithm on pre-processed data
```python

community_topic = CommunityTopic(train_corpus=tokenized_bbc_train_sents,  dictionary=dictionary)
community_topic.fit()
```

> Step 5: Get topic words founded by abovr algorithm
```python

topic_words = community_topic.get_topics_words_topn(10)
```

## API Usage

Following are the API functions that we expose by this library code:

| Method | Code |
|---|---|
| Fit the flat topic model | .fit() |
| Fit the hiearchical topic model | .fit_hierarchical() |
| Get flat topic words | .get_topics_words() |
| Get topn _n_ flat topic word | .get_topics_words_topn(n=10) |
| Get flat topics as dictionary id | .get_topics() |
| Get hierarchical topic words | .get_topic_words_hierarchical() |
| Get hierarchical topic as dictionary id an ig_graph of that topic | .get_topics_hierarchical() |
| Geet first _n_ levels in hierarchy | .get_n_level_topic_words_hierarchical(n_level=2) |
| Geet hierarchical topic words in a tree-like dictionary format | .get_hierarchy_tree |

Note for more detailed information go to [website documentation](https://shr1911.github.io/communitytopic/)


