# https://stackoverflow.com/questions/51903172/how-to-display-a-tree-in-python-similar-to-msdos-tree-command/51920869#51920869

from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time

import pickle

from collections import defaultdict
import pprint


def get_input_data(community_topic):
    input_data = {}

    for i in range(1, community_topic.get_num_levels_count() + 1):
        dictionary = community_topic.get_nth_level(i)
        for key in dictionary:
            input_data[key] = ' '.join(dictionary[key])
    return input_data


def add_node(tree, path, value):
    """
    Adds a node to the tree structure at the specified path with the specified value.
    """
    node = tree
    l = len(path) - 1
    for i, key in enumerate(path):
        if i == l:
            node.update({key: {}})
        else:
            node = node[key]
        i_key = key
    node[i_key]['value'] = value


def construct_tree(dict_data):
    """
    Constructs a tree structure from the given dictionary data.
    """
    tree = defaultdict(dict)
    for key, value in dict_data.items():
        path = key.split('_')
        add_node(tree, path, value)
    return tree


def main():
    # with open("../../text_datasets/bbc_train.txt", "r") as f:
    #     bbc_train = f.read()
    # with open("../../text_datasets/bbc_test.txt", "r") as f:
    #     bbc_test = f.read()
    #
    # tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(bbc_train, bbc_test, ner=1,
    #                                                                          pos_filter=0,
    #                                                                          phrases="npmi",
    #                                                                          phrase_threshold=0.35)
    #
    # with open("../tokenized_bbc_train_sents.obj", "wb") as f:
    #     pickle.dump(tokenized_bbc_train_sents, f)
    #
    # with open("../tokenized_bbc_train_docs.obj", "wb") as f:
    #     pickle.dump(tokenized_bbc_train_docs, f)
    #
    # with open("../tokenized_bbc_test_docs.obj", "wb") as f:
    #     pickle.dump(tokenized_bbc_test_docs, f)
    #
    # with open("../dictionary.obj", "wb") as f:
    #     pickle.dump(dictionary, f)

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

    community_topic.fit_hierarchical(2)
    t1 = time()
    topic_words = community_topic.get_n_level_topic_words_hierarchical(2)

    # input_data = get_input_data(community_topic)
    # tree = construct_tree(input_data)
    # print(tree)

    community_topic.get_hierarchy_tree()


    print(f"community topic on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
