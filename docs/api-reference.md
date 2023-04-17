# API Usage

There are two important classes in the library as follows:-
- PreProcessing
- CommunityTopic

We will see API usage of both of them.


## Pre-Processing Module

### <u>Method: `do_preprocessing`</u>

The `do_preprocessing` method performs pre-processing on a given training and testing corpus to convert it into a format suitable for `CommunityTopic`.

#### Parameters

```python
do_preprocessing(train=None, test=None, ner=1, pos_filter=0, phrases='npmi', phrase_threshold=0.35, language='en')
```

- `train` : `str`

    Input training corpus


- `test` : `str`

    Input testing corpus


- `ner` : `int`

    - Named Entity Recognition flag 
    - Possible values = [0, 1]
    - 0 - to not use NER 
    - 1 - to use NER


- `pos_filter ` : `int`

    - Part-of-Speech filter for extracting features and marking the words in a text with labels for entity extraction
    - Possible values = [0, 1, 2, 3]
    - 0 - No POS filtering 
    - 1 - Keep only adjectives, adverbs, nouns, proper nouns, and verbs 
    - 2 - Keep only adjectives, nouns, proper nouns 
    - 3 - Keep only nouns, proper nouns


- `phrases` : `str`

    Currently using 'npmi' type for phrase detection


- `phrase_threshold` : `float`

    - Phrase detection threshold 
    - Currently using 0.35


- `language` : `str`

    - Possible values = ['en', 'it', 'fr', 'de', 'es'] 
    - 'en' - English
    - 'it' - Italian
    - 'fr' - French
    - 'de' - German
    - 'es' - Spanish
    - Language of the training and testing corpus

#### Returns

```python
tokenized_train_sents
tokenized_train_docs
tokenized_test_docs
dictionary
```

- `tokenized_train_sents` : list of list

    Returns pre-processed training corpus as sentences (in list of words form)


- `tokenized_train_docs` : list of list

    Returns pre-processed training corpus as docs (in list of words form)


- `tokenized_test_docs` : list of list

    Returns pre-processed testing corpus as sentences (in list of words form)


- `dictionary` : dict

    - Gensim dictionary object that tracks frequencies and can filter vocab
    - Keys are id for words
    - Values are words

<hr style="border: 2px solid grey">

## Community Topic Module

###  <u>Class constructor: `__init__` </u>
```python
__init__(self, train_corpus=None, dictionary=None, edge_weight='count',
         weight_threshold=0.0, cd_algorithm='leiden', resolution_parameter=1.0,
         network_window='sentence')
```

#### Parameters

- `train_corpus` : `list of list`  (of string)

    - Preprocessed sentences of training corpus (List of list)
    - It contains pre-processed tokenized sentence as list of list


- `dictionary` : `dict`

    - Gensim dictionary object that tracks frequencies and can filter vocab 
    - keys are id for words 
    - values are words


- `edge_weight`: `str`

    - It is weight of edges which comes from the frequency of co-occurrence. 
    - Possible values: ["count", "npmi"]
    - "count":  Raw count of possible edges as the edge weight.
    - "npmi": Weighing scheme which uses Normalized Pointwise Mutual Information (NPMI) between terms


- `weight_threshold` : `float`

    - The edges can be thresholded, i.e. those edges whose weights fall below a
       certain threshold are removed from the network.


- `cd_algorithm` : `str`

    - To choose community detection algorithm, possible values: ["leiden", "walktrap"]


- `resolution_parameter` : `float`

    - Te resolution_parameter to use for leiden community detection algorithm. 
    - Higher resolution_parameter lead to smaller communities, while 
    - lower resolution_parameter lead to fewer larger communities.


- `network_window`:

    - The network that we construct from a corpus has terms as vertices. This decides the
           fixed sliding window of document. 
    - Possible values: ["sentence", "5", "10"]
    - "sentence": two terms co-occur if they both occur in the same sentence.
    - "5" or "10": two terms co-occur if they both occur within a fixed-size sliding window over a document.

<hr style="border: 2px solid grey">


###  <u>Method: `fit` </u>
```python
fit()
```

This method performs task of finding simple topics

<hr style="border: 2px solid grey">


###  <u>Method: `fit_hierarchical` </u>
```python
fit_hierarchical(n_level=2)
```

This method performs task of finding hierarchical topics
 

#### Parameter
- `n_level` : `int`

    - Number of level for hierarchical topics
 
<hr style="border: 2px solid grey">


###  <u>Method: `get_topics_words` </u>
```python
get_topics_words()
```

Get topic words of flat topic modelling

#### Returns
- `topics` : `list of list`

    - Returns flat topics as topic words

<hr style="border: 2px solid grey">


###  <u>Method: `get_topics_words_topn` </u>
```python
get_topics_words_topn(n=10)
```

Get top n topic words of flat topic modelling

#### Parameter
- `n` : `int`

    - top n topic words

#### Returns
- `topics` : `list of list`

    - Returns top n flat topics as topic words

<hr style="border: 2px solid grey">


###  <u>Method: `get_topics` </u>
```python
get_topics()
```

Get topic as dictionary id


#### Returns
- `topics` : `list of list`

    - Returns flat topics as dictionary id

<hr style="border: 2px solid grey">


###  <u>Method: `get_topic_words_hierarchical` </u>
```python
get_topic_words_hierarchical()
```

Get hierarchical topic as topic words


#### Returns
- `hierarchical_topics_words` : `dict of dict`

    - In following format (each level and topic in that level)-

        ```
        { 1 :   {"0": ['firm', 'company', 'economy',...],
                 "1": ['country', 'china', 'bank'....], }
          .....},
          2 :   {"0": [''orders', 'spring', 'allies',...],
                 "1": ['lawyer', 'individuals', 'failure'....], }
          .....},
          .....
        }
        ```
      
<hr style="border: 2px solid grey">

    
###  <u>Method: `get_topics_hierarchical` </u>
```python
get_topics_hierarchical()
```

Get hierarchical topic as dictionary id, and ig_graph of topic


#### Parameter
- `n` : `int`

    - top n topic words

#### Returns
- `hierarchical_topics` : `dict of dict`

    - Returns top hierarchical topics as topic words
    - In following format (each level and topic in that level)-

      ```
      {
      1 : {"0": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph },
            "1": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph },
            .....},
      2 : {"0_0": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph},
           "0_1": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph},
           .....
           .....          
           "1_0": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph},
           "1_1": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph},
           .....},
      3 : {"0_0_0": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph},
           "0_0_1": {'dict_num': [2, 147, 6, 1180, 327, ,....], 'ig_graph': object of ig_graph},
          .....
          },
      ......
      }
      ```
 
      Note, in above format each level has topic names as the key of dictionary.
          For example, level 1 has single digit value which specifies topics in that level
          level 2 has two values seperated by underscore, first value is super topic and second is child topic
          Similary, level 3 has three values, for which parent topics and current child topic

<hr style="border: 2px solid grey">


###  <u>Method: `get_topics_hierarchical` </u>
```python
get_n_level_topic_words_hierarchical(n_level=2)
```

Get first n number of levels from hierarchy

#### Parameter
- `n_level` : `int`

    - top n level

#### Returns
- `topics` : `dict of dict`

<hr style="border: 2px solid grey">



###  <u>Method: `get_hierarchy_tree` </u>
```python
get_hierarchy_tree()
```

This function is for visualisation purpose of hierarchical topics.

#### Returns
- `tree` : It returns a tree-like structure in dictionary format.

