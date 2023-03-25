import gensim

from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time
import pickle
from diversity_metrics import *
import fasttext

from gensim.test.utils import datapath
from gensim.models import KeyedVectors





def main():
    # with open("./tokenized_train_corpus_docs.obj", "rb") as f:
    #     tokenized_train_corpus_docs = pickle.load(f)
    #
    # with open("./dictionary.obj", "rb") as f:
    #     dictionary = pickle.load(f)
    #
    # with open("./tokenized_test_corpus_docs.obj", "rb") as f:
    #     tokenized_test_corpus_docs = pickle.load(f)


    # with open("../tokenized_bbc_train_sents.obj", "rb") as f:
    #     tokenized_bbc_train_sents = pickle.load(f)
    #
    # with open("../tokenized_bbc_train_docs.obj", "rb") as f:
    #     tokenized_bbc_train_docs = pickle.load(f)
    #
    # with open("../tokenized_bbc_test_docs.obj", "rb") as f:
    #     tokenized_bbc_test_docs = pickle.load(f)
    #
    # with open("../dictionary.obj", "rb") as f:
    #     dictionary = pickle.load(f)

    with open("../../../text_datasets/reuters_train.txt", "r") as f:
        bbc_train = f.read();
    with open("../../../text_datasets/reuters_test.txt", "r") as f:
        bbc_test = f.read();

    tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
            bbc_train, bbc_test, ner=1,
            pos_filter=3,
            phrases="npmi",
            phrase_threshold=0.35,
            language="en")

    t0 = time()
    community_topic = CommunityTopic(corpus=tokenized_bbc_train_sents,
                                     dictionary=dictionary,
                                     edge_weight="npmi",
                                     weight_threshold=0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1,
                                     network_window="sentence")

    community_topic.fit()
    t1 = time()

    topic_words = community_topic.get_topics_words()
    print(topic_words)
    print(f"Num topics: {len(topic_words)}")

    print("puw:", proportion_unique_words(topic_words, topk=10))
    print("jd:", pairwise_jaccard_diversity(topic_words, topk=10))
    print("irbo p=0.5:", irbo(topic_words, weight=0.5, topk=10))
    print("irbo p=0.9:", irbo(topic_words, weight=0.9, topk=10))

    # for coherence in ["c_v", "c_npmi"]:
    #     for topn in [5, 10, 20]:
    #         cm = CoherenceModel(topics=topic_words,
    #                             texts=tokenized_bbc_test_docs,
    #                             dictionary=dictionary,
    #                             topn=topn,
    #                             coherence=coherence)
    #         score = cm.get_coherence()
    #         print(coherence,":",score,"(topn=",topn,")")

    # cm = CoherenceModel(topics=topic_words,
    #                     texts=tokenized_test_corpus_docs,
    #                     dictionary=dictionary,
    #                     topn=5,
    #                     coherence="c_v")
    # score = cm.get_coherence()
    # print(score)
    print(f"community topic test on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
