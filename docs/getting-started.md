## Getting Started

This is an example tuotrial which finds topic of BBC dataset using best combination for Pre-Processing and Community Topic Algorithm. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1npBdhbDI7c3NOfjhgbLYvUba6bLjAyUu?usp=sharing)

> Step 1: import necessary class of the library
```python
from communitytopic import CommunityTopic
from communitytopic import PreProcessing
```

> Step 2: Load raw corpus as the dataset, here we are using BBC dataset. 
```python
with open("<Path-To-train-Dataset>/bbc_train.txt", "r", encoding='utf-8') as f:
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