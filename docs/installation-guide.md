## Installation guide

- System requirement

  ```
      Python >= 3.6
      commodity hardware
      setuptools~=67.6.0
      spacy~=3.5.0
      numpy~=1.21.5
      gensim~=4.2.0
      networkx~=2.8.4
      igraph~=0.10.4
  ```

- Installation Option

    The easy way to install CommunityTopic is:
     ```
      pip install communitytopic
    ```
   
   
- Spacy models for different languages for pre-processing (Commuity Topic has a pre-processing function as shown in below getting started example, and it requires spacy model to be dowloaded via python for the language which we are using). Following are commands for the same :

| Language | Commond to download spacy model |
|---|---|
| English | !python -m spacy download en_core_web_sm |
| Italian | !python -m spacy download it_core_news_sm |
| French | !python -m spacy download fr_core_news_sm |
| German | !python -m spacy download de_core_news_sm |
| Spanish | !python -m spacy download es_core_news_sm |