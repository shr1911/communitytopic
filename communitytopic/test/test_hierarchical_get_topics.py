from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time

import pickle


def main():
    # with open("../text_datasets/bbc_train.txt", "r") as f:
    #     bbc_train = f.read();
    # with open("../text_datasets/bbc_test.txt", "r") as f:
    #     bbc_test = f.read();
    #
    # tokenized_train_corpus_docs, dictionary = PreProcessing.do_preprocessing(bbc_train, ner=1,
    #                                                                          pos_filter=0,
    #                                                                          phrases="npmi",
    #                                                                          phrase_threshold=0.35)
    #
    # with open("./tokenized_train_corpus_docs.obj", "wb") as f:
    #     pickle.dump(tokenized_train_corpus_docs, f)
    # with open("./dictionary.obj", "wb") as f:
    #     pickle.dump(dictionary, f)
    #
    # tokenized_test_corpus_docs, test_dictionary = PreProcessing.do_preprocessing(bbc_test, ner=1,
    #                                                                              pos_filter=0,
    #                                                                              phrases="npmi",
    #                                                                              phrase_threshold=0.35)
    # with open("./tokenized_test_corpus_docs.obj", "wb") as f:
    #     pickle.dump(tokenized_test_corpus_docs, f)

    with open("./tokenized_train_corpus_docs.obj", "rb") as f:
        tokenized_train_corpus_docs = pickle.load(f)

    with open("./dictionary.obj", "rb") as f:
        dictionary = pickle.load(f)

    with open("./tokenized_test_corpus_docs.obj", "rb") as f:
        tokenized_test_corpus_docs = pickle.load(f)

    t0 = time()
    community_topic = CommunityTopic(corpus=tokenized_train_corpus_docs,
                                     dictionary=dictionary,
                                     edge_weight="count",
                                     weight_threshold=0.0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1.0,
                                     network_window="sentence")

    community_topic.fit_hierarchical(2)
    t1 = time()

    topic_words = community_topic.get_n_topic_words_hierarchical(2)
    print(topic_words[2])
    print(len(topic_words[2]))

    # # topic_words = community_topic.get_hierarchical_topics()
    # for i, level in topic_words.items():
    #     if i == 0:
    #         pass
    #     else:
    #         print("printing level" + str(i))
    #         print(level)
    #         print("--------------------------------")

    print(f"community topic on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
