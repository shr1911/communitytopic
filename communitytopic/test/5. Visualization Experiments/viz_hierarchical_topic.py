# https://stackoverflow.com/questions/51903172/how-to-display-a-tree-in-python-similar-to-msdos-tree-command/51920869#51920869

from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time

import pickle


def ptree(start, tree, indent_width=4):

    def _ptree(start, parent, tree, grandpa=None, indent=""):
        if parent != start:
            if grandpa is None:  # Ask grandpa kids!
                print(parent, end="")
            else:
                print(parent)
        if parent not in tree:
            return
        for child in tree[parent][:-1]:
            print(indent + "├" + "─" * indent_width, end="")
            _ptree(start, child, tree, parent, indent + "│" + " " * 4)
        child = tree[parent][-1]
        print(indent + "└" + "─" * indent_width, end="")
        _ptree(start, child, tree, parent, indent + " " * 5)  # 4 -> 5

    parent = start
    print(start)
    _ptree(start, parent, tree)




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

    with open("../tokenized_bbc_train_sents.obj", "rb") as f:
        tokenized_bbc_train_sents = pickle.load(f)

    with open("../tokenized_bbc_train_docs.obj", "rb") as f:
        tokenized_bbc_train_docs = pickle.load(f)

    with open("../tokenized_bbc_test_docs.obj", "rb") as f:
        tokenized_bbc_test_docs = pickle.load(f)

    with open("../dictionary.obj", "rb") as f:
        dictionary = pickle.load(f)


    t0 = time()
    community_topic = CommunityTopic(train_corpus=tokenized_bbc_train_sents,
                                     dictionary=dictionary,
                                     edge_weight="count",
                                     weight_threshold=0.0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1.0,
                                     network_window="sentence")

    community_topic.fit_hierarchical(3)
    t1 = time()
    topic_words = community_topic.get_n_level_topic_words_hierarchical(3)
    level_1 = community_topic.get_nth_level(1)
    print(community_topic.get_nth_level(1)) # l1 = length of level 1 (loop till l-1)
    print(community_topic.get_nth_level(2))
    print(community_topic.get_nth_level(3))

    '''
    -1: level_1
    '''


    # dct = {
    #     -1: [0, 60000],
    #     0: [100, 20, 10],
    #     100: [30],
    #     30: [400, 500],
    #     60000: [70, 80, 600],
    #     500: [495, 496, 497]
    # }
    # ptree(-1, dct)


    print(f"community topic on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
