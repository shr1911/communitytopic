import re

import spacy
from spacy.tokens import Token, Doc, Span
from spacy.language import Language
from spacy.attrs import LOWER, SENT_START
from typing import Union, Iterable, Iterator, List, Optional, Set, Dict, Tuple
import math
import re
import numpy as np
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.corpora import Dictionary
from copy import deepcopy
from time import time

# Following are connector words for different languages which is used for phrase detection task in pre-processing
ITALIAN_CONNECTOR_WORDS = ['e', 'ed', 'da', 'o', 'da', 'per', 'con', 'a', 'an', 'il', 'presso', 'in', 'senza', 'o', 'a',
                           'su', 'oppure']
FRENCH_CONNECTOR_WORDS = ["le", "et", "à", "à", "ou", "or", "avec", "de", "pour", "ou", "sans", "par", "sur", "dans",
                          "un", "une"]
GERMAN_CONNECTOR_WORDS = ['der', 'und', 'bei', 'zu', 'oder', 'mit', 'von', 'für', 'von', 'ohne', 'durch', 'am', 'in',
                          'an', 'a']
SPANISH_CONNECTOR_WORDS = ['el', 'y', 'en', 'a', 'o', 'con', 'de', 'para', 'de', 'sin', 'por', 'sobre', 'en', 'un', 'a']


def do_preprocessing(train=None, test=None, ner=1, pos_filter=0, phrases="npmi", phrase_threshold=0.35, language="en"):
    """
    This method perform pre-processing on train and test corpus, which makes it in the form needed by CommunityTopic

    @param train: String
        input training corpus

    @param test: String
        input testing corpus

    @param ner: int
        Named Entity Recogition flag
        Possible values = [0, 1]
        0 - to not use NER
        1 - to use NER

    @param pos_filter: int
        Part-of-Speech filter is (entity extraction) for extracting features and marks the word in a text with labels
        Possible values = [0, 1, 2, 3]
        0 - No POS filtering
        1 - Keep only adjectives, adverbs, nouns, proper nouns, and verbs
        2 - Keep only adjectives, nouns, proper nouns
        3 - Keep only nouns, proper nouns

    @param phrases: String
        'npmi' - currently using 'npmi' type for phrase detection

    @param phrase_threshold: float
        Phrase detection threshold
        Currently using 0.35

    @param language: String
        Possible values = ['en', 'it', 'fr', 'de', 'es']
        'en' - English
        'it' - Italian
        'fr' - French
        'de' - German
        'es' - Spanish
        Language of the training and testing corpus

    @return tokenized_train_sents: list of list
        Returns pre-processed training corpus as sentence (in list of words form)

    @return tokenized_train_docs: list of list
        Returns pre-processed training corpus as docs (in list of words form)

    @return tokenized_test_docs: list of list
        Returns pre-processed training corpus as sentence (in list of words form)

    @return dictionary: dict
        Gensim dictionary object that tracks frequencies and can filter vocab
        - keys are id for words
        - values are words

    """
    train = train.split("\n")
    test = test.split("\n")

    # need to specify entity types for NER
    ent_types = ["EVENT", "FAC", "GPE", "LOC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]
    if ner == 0:
        use_ner = False
    else:
        use_ner = True

    pos_types = None
    if pos_filter == 0:
        pos_types = None
    # filter 1 means we want to keep adjectives, adverbs, nouns, proper nouns, and verbs
    elif pos_filter == 1:
        pos_types = ["ADJ", "ADV", "NOUN", "PROPN", "VERB"]
    # filter 2 eliminates adverbs and verbs
    elif pos_filter == 2:
        pos_types = ["ADJ", "NOUN", "PROPN"]
    # filter 3 is just nouns and proper nouns
    elif pos_filter == 3:
        pos_types = ["NOUN", "PROPN"]

    # create filter configuration dict
    filter_dict = {"filter_short": True,
                   "filter_stopwords": True,
                   "filter_numbers": True,
                   "filter_punct": True,
                   "filter_websites": True,
                   "filter_emails": True,
                   "filter_not_wordlike": True,
                   "pos_filters": pos_types}

    spacy_model = "en_core_web_sm"
    if language == 'it':
        spacy_model = "it_core_news_sm"
    elif language == 'fr':
        spacy_model = "fr_core_news_sm"
    elif language == 'de':
        spacy_model = 'de_core_news_sm'
    elif language == 'es':
        spacy_model = 'es_core_news_sm'
    else:
        pass

    # create preprocessing pipeline
    nlp = create_pipeline(spacy_model=spacy_model,
                          detect_sentences=True,
                          detect_entities=use_ner,
                          entity_types=ent_types,
                          filter_config=filter_dict)

    print("Preprocessing documents...")
    t0 = time()
    train_docs = list(nlp.pipe(train))
    test_docs = list(nlp.pipe(test))

    tokenized_train_docs = list(tokenize_docs(train_docs, lowercase=True, sentences=False))
    tokenized_train_sents = list(tokenize_docs(train_docs, lowercase=True, sentences=True))
    tokenized_test_docs = list(tokenize_docs(test_docs, lowercase=True, sentences=False))

    phrases, tokenized_train_docs, phrase_models = detect_phrases(tokenized_train_docs,
                                                                  num_iterations=2,
                                                                  scoring_method='npmi',
                                                                  threshold=0.35,
                                                                  min_count=None,
                                                                  language=language)

    for model in phrase_models:
        tokenized_train_sents = model[tokenized_train_sents]
    tokenized_train_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_train_sents]

    for model in phrase_models:
        tokenized_test_docs = model[tokenized_test_docs]
    tokenized_test_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_test_docs]

    vocab, dictionary = create_vocabulary_and_dictionary(tokenized_train_docs, min_threshold=None)
    tokenized_train_sents = filter_tokenized_docs_with_vocab(tokenized_train_sents, vocab)
    tokenized_train_docs = filter_tokenized_docs_with_vocab(tokenized_train_docs, vocab)
    tokenized_test_docs = filter_tokenized_docs_with_vocab(tokenized_test_docs, vocab)
    test_vocab = set()
    for doc in tokenized_test_docs:
        for token in doc:
            test_vocab.add(token)
    tokenized_train_docs = [[token for token in doc if token in test_vocab] for doc in tokenized_train_docs]
    tokenized_train_sents = [[token for token in sent if token in test_vocab] for sent in tokenized_train_sents]

    tokenized_train_sents = [sent for sent in tokenized_train_sents if len(sent) > 0]
    tokenized_train_docs = [doc for doc in tokenized_train_docs if len(doc) > 0]
    tokenized_test_docs = [doc for doc in tokenized_test_docs if len(doc) > 0]

    # if not (phrases == "none"):
    #     phrases, tokenized_docs, phrase_models = detect_phrases(tokenized_docs,
    #                                                             num_iterations=2,
    #                                                             scoring_method=phrases,
    #                                                             threshold=phrase_threshold,
    #                                                             min_count=None)
    #     for model in phrase_models:
    #         tokenized_sents = model[tokenized_sents]
    #     tokenized_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_sents]
    # else:
    #     tokenized_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_docs]
    #     tokenized_sents = [[token.replace(" ", "_") for token in sent] for sent in tokenized_sents]
    #
    # vocab, dictionary = create_vocabulary_and_dictionary(tokenized_docs, min_threshold=None)
    # tokenized_sents = filter_tokenized_docs_with_vocab(tokenized_sents, vocab)
    # tokenized_corpus_docs = [sent for sent in tokenized_sents if len(sent) > 0]
    t1 = time()
    print(f"Preprocessing completed in {t1 - t0} seconds")
    return tokenized_train_sents, tokenized_train_docs, tokenized_test_docs, dictionary


class EntityMergerPipe:
    """
    Custom entity merger pipe so that we can filter entity types and only merge those of specified type
    """

    def __init__(
            self,
            entity_types: Optional[Iterable[str]],
    ) -> None:
        self.entity_types = set(entity_types) if entity_types is not None else entity_types

    def __call__(
            self,
            doc: Doc
    ) -> Doc:
        with doc.retokenize() as retokenizer:
            for entity in doc.ents:
                if self.entity_types is None or entity.label_ in self.entity_types:
                    attrs = {"tag": entity.root.tag,
                             "dep": entity.root.dep,
                             "ent_type": entity.label}
                    while entity and entity[0].is_stop:
                        entity = entity[1:]
                    if entity:
                        retokenizer.merge(entity, attrs=attrs)
        return doc


@Language.factory("custom_entity_merger", default_config={"entity_types": None})
def create_entity_merger(
        nlp: Language,
        name: str,
        entity_types: Optional[Iterable[str]]
) -> EntityMergerPipe:
    return EntityMergerPipe(entity_types)


class TokenFilterPipe:
    """
    Custom pipe to remove tokens from doc based on set of filters passed by user.
    Tokens are actually removed from doc but import information is retained from
    the processing of full docs e.g. parts of speech, entities, sentences.
    """

    def __init__(
            self,
            detect_sentences: bool,
            filter_dict: Dict[str, Union[bool, Optional[Iterable[str]]]]
    ) -> None:
        self.detect_sentences = detect_sentences
        self.filter_short = filter_dict.get("filter_short", False)
        self.filter_stopwords = filter_dict.get("filter_stopwords", False)
        self.filter_numbers = filter_dict.get("filter_numbers", False)
        self.filter_punct = filter_dict.get("filter_punct", False)
        self.filter_websites = filter_dict.get("filter_websites", False)
        self.filter_emails = filter_dict.get("filter_emails", False)
        self.filter_not_wordlike = filter_dict.get("filter_not_wordlike", False)
        if self.filter_not_wordlike:
            self.regex = re.compile(r"^[\w-]*[a-zA-Z][\w-]*$")
        self.pos_filters = None
        if filter_dict.get("pos_filters", None) is not None:
            self.pos_filters = set(filter_dict["pos_filters"])
        self.attrs = [LOWER]
        if self.detect_sentences:
            self.attrs = self.attrs + [SENT_START]

    def __call__(
            self,
            doc: Doc,
    ) -> Doc:
        # get the indices to remove i.e. those tokens that don't pass some filter
        indices_to_remove = set()
        for token in doc:
            if self.remove_token(token):
                indices_to_remove.add(token.i)
        # we are going to split up the document to filter tokens, but need to keep some
        # token attributes that can't be properly parsed from the filtered tokens
        doc_array = doc.to_array(self.attrs)
        num_tokens = doc_array.shape[0]
        # if we have detected sentences we need to move sentence starts past any filtered tokens
        if self.detect_sentences:
            start_indices = set((doc_array[:, -1] == 1).nonzero()[0])
            # while some start indices are marked for removal move those up by one position
            starts_to_change = indices_to_remove.intersection(start_indices)
            while starts_to_change:
                for index in starts_to_change:
                    start_indices.remove(index)
                    # if start is at end of doc i.e. entire sentence was removed, then don't add
                    if (index + 1) < num_tokens:
                        start_indices.add(index + 1)
                starts_to_change = indices_to_remove.intersection(start_indices)
            # now we set the appropriate indices to be sentence starts
            doc_array[list(start_indices), -1] = 1
        # delete rows corresponding to filtered tokens
        doc_array = np.delete(doc_array, list(indices_to_remove), axis=0)
        new_doc = Doc(doc.vocab, words=[t.text for t in doc if t.i not in indices_to_remove])
        new_doc.from_array(self.attrs, doc_array)
        return new_doc

    def remove_token(self, token: Token) -> bool:
        """
        Determine whether to remove token from doc based on filter arguments.
        """
        if self.filter_short and token.__len__() < 3:
            return True
        if self.filter_stopwords and token.is_stop:
            return True
        if self.filter_numbers and token.like_num:
            return True
        if self.filter_punct and token.is_punct:
            return True
        if self.filter_websites and token.like_url:
            return True
        if self.filter_emails and token.like_email:
            return True
        if self.pos_filters is not None and token.pos_ not in self.pos_filters:
            return True
        if self.filter_not_wordlike:
            text = token.text.replace(" ", "_")
            if self.regex.search(text) is None:
                return True
        return False


@Language.factory("custom_token_filter")
def create_token_filter(
        nlp: Language,
        name: str,
        detect_sentences: bool,
        filter_dict: Dict[str, Union[bool, Optional[Iterable[str]]]]
) -> TokenFilterPipe:
    return TokenFilterPipe(detect_sentences, filter_dict)


def create_pipeline(
        spacy_model: str = "en_core_web_sm",
        detect_sentences: bool = False,
        detect_entities: bool = False,
        entity_types: Optional[Iterable[str]] = None,
        filter_config: Optional[Dict[str, bool]] = None,
) -> Language:
    """
    Create and return a spacy language pipeline configured as specified by passed arguments.
    """
    # load model and disable Named Entity Recognition and lemmatizing to speed things up if we
    # don't need them
    # all other pipes will be needed for almost every other case we consider so leave them active
    nlp = spacy.load(spacy_model,
                     disable=["lemmatizer", "ner"])
    if detect_sentences:
        nlp.enable_pipe("senter")
    if detect_entities:
        nlp.enable_pipe("ner")
    if detect_entities:
        nlp.add_pipe("custom_entity_merger",
                     config={"entity_types": entity_types},
                     after="ner")
    if filter_config is not None:
        nlp.add_pipe("custom_token_filter",
                     config={"detect_sentences": detect_sentences,
                             "filter_dict": filter_config})
    return nlp


def tokenize_docs(
        docs: Iterable[Doc],
        lowercase: bool = True,
        sentences: bool = False,
) -> Iterator[List[str]]:
    for doc in docs:
        if sentences:
            for sentence in doc.sents:
                yield [t.lower_ if lowercase else t.text for t in sentence]
        else:
            yield [t.lower_ if lowercase else t.text for t in doc]


def detect_phrases(
        tokenized_docs: Iterable[List[str]],
        num_iterations: int = 1,
        scoring_method: str = "default",
        threshold: float = 1.0,
        min_count: Optional[int] = None,
        language: str = "en"
) -> Tuple[Set[str], List[List[str]], List[Phrases]]:
    """
    Use gensim to detect and return meaningful n-grams. Return the n-grams, the docs with n-grams
    merged, and a list of phrase models to be able to merge other docs e.g. the sentences.
    """
    # if no min count is provided we will set it to be the same as default min
    # doc count for terms because otherwise it will be excluded anyway
    if min_count is None:
        min_count = math.ceil(2 * (0.02 * len(tokenized_docs)) ** (1 / math.log(10, math.e)))
    phrases = set()
    phrase_models = []
    connector_words = ENGLISH_CONNECTOR_WORDS
    if language == 'it':
        connector_words = ITALIAN_CONNECTOR_WORDS
    elif language == 'fr':
        connector_words = FRENCH_CONNECTOR_WORDS
    elif language == 'de':
        connector_words = GERMAN_CONNECTOR_WORDS
    elif language == 'es':
        connector_words = SPANISH_CONNECTOR_WORDS
    else:
        pass

    for iteration in range(num_iterations):
        # detect significant bi-grams in text
        phrase_model = Phrases(tokenized_docs,
                               min_count=min_count,
                               threshold=threshold,
                               scoring=scoring_method,
                               connector_words=connector_words)
        # add detected bi-grams to set
        phrases |= phrase_model.export_phrases().keys()
        # add copy of phrase model to list
        phrase_models.append(deepcopy(phrase_model))
        # modify texts to merge detected n-grams together
        tokenized_docs = phrase_model[tokenized_docs]
    # remove underscores connecting words in n-grams
    tokenized_docs = [[token.replace(" ", "_") for token in doc] for doc in tokenized_docs]
    return phrases, tokenized_docs, phrase_models


def create_vocabulary_and_dictionary(
        texts: Iterable[List[str]],
        max_threshold: float = 0.90,
        min_threshold: Optional[float] = None
) -> Tuple[Set[str], Dictionary]:
    """
    Create a gensim dictionary object that tracks frequencies and can filter vocab.
    If min threshold is set to None then will be automatically calculated based on corpora size.
    If you want not min/max filtering then use 0.0/100.0 as values.
    """
    # create and "load" dictionary
    dictionary = Dictionary(texts)
    _ = dictionary[0]
    # if specific min threshold not specified calculate based on corpus size
    if min_threshold is None:
        min_docs = math.ceil(2 * (0.02 * dictionary.num_docs) ** (1 / math.log(10, math.e)))
    else:
        min_docs = math.ceil(dictionary.num_docs * min_threshold)
    # filter extremes
    dictionary.filter_extremes(min_docs, max_threshold)
    # create vocab set and return both vocab and dictionary
    vocab = set(dictionary.token2id.keys())

    return vocab, dictionary


def filter_tokenized_docs_with_vocab(
        tokenized_docs: Iterable[List[str]],
        vocab: Set[str]
) -> List[List[str]]:
    filtered_docs = []
    for doc in tokenized_docs:
        filtered_tokens = []
        for token in doc:
            if token in vocab:
                filtered_tokens.append(token)
        filtered_docs.append(filtered_tokens)
    return filtered_docs
