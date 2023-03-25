from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time
import pickle
import numpy as np
from typing import Union, Iterable, Iterator, List, Optional, Set, Dict, Tuple
import networkx as nx
from diversity_metrics import *


def get_internal_weighted_degree(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the internal weighted degree of node corresponding to term,
    i.e. the sum of edge strengths connecting to nodes in same community
    """
    community_subgraph = graph.subgraph(community)
    return community_subgraph.degree(weight="weight")[str(index)]


def get_topic_phi(topic, nx_g, level_0):
    c = [str(node) for node in topic]
    phi = np.zeros((len(level_0),))
    for i, v in enumerate(level_0):
        if v in topic:
            phi[i] = get_internal_weighted_degree(v, c, nx_g)

    return phi


def main():
    """
    This is a hierarchical topic evaluation which contains the best combination for pre-processing and CommunityTopic Algorith.

    1. Pre-processes training and testing corpus
    2. Apply CommunityTopic Algorithm
    3. Get topic words
    4. Calculate coherence score ('c_v', 'c_npmi', 'u_mass') for testing data
    5. Calculate level 1, level 2, level 3 specialization
    6. Calculate affinity between level 1 and level 2

    """
    with open("../../../text_datasets/europarl_en_train.txt", "r", encoding='utf-8') as f:
        bbc_train = f.read()
    with open("../../../text_datasets/europarl_en_test.txt", "r", encoding='utf-8') as f:
        bbc_test = f.read()

    tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
        train=bbc_train,
        test=bbc_test,
        ner=1,
        pos_filter=3,
        phrases="npmi",
        phrase_threshold=0.35,
        language="en")

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

    t0 = time()
    community_topic = CommunityTopic(train_corpus=tokenized_bbc_train_sents,
                                     dictionary=dictionary,
                                     edge_weight="npmi",
                                     weight_threshold=0.0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1.0,
                                     network_window="sentence")

    community_topic.fit_hierarchical(2)
    t1 = time()

    topic_words = community_topic.get_n_level_topic_words_hierarchical(2)
    print(topic_words)

    eval_topics = []
    for key, level in topic_words.items():
        for i, topic in level.items():
            eval_topics.append(topic)

    print("puw:", proportion_unique_words(eval_topics, topk=10))


    # Coherence scores for hierarchical
    for coherence in ["c_v", "c_npmi", "u_mass"]:
        cm = CoherenceModel(topics=eval_topics, texts=tokenized_bbc_test_docs, dictionary=dictionary, topn=5,
                            coherence=coherence)
        score = cm.get_coherence()
        print(f'{coherence}: {score}')

    # Data needed to calculate specialization
    level_0 = community_topic.get_root_level_0()
    topics = community_topic.get_topic_hierarchical()
    phi_norm = np.array([community_topic.sentence_nb.token_freqs[v] for v in level_0])
    phi_norm = phi_norm / phi_norm.sum()

    # Topic Specialization of level 1
    level = topics[1]
    print("level 1 topics: ", len(level))

    phis = []
    for t in level.values():
        topic = t["dict_num"]
        phi = get_topic_phi(topic, community_topic.nx_g, level_0)
        phi = phi / phi.sum()
        cos = np.dot(phi, phi_norm) / (np.linalg.norm(phi) * np.linalg.norm(phi_norm))
        phis.append(1 - cos)
    print("Avg specialization level 1")
    print(sum(phis) / len(phis))

    # Topic Specialization for level 2
    level = topics[2]
    print("level 2 topics: ", len(level))
    phis = []

    for t in level.values():
        topic = t["dict_num"]
        phi = get_topic_phi(topic, community_topic.nx_g, level_0)
        phi = phi / phi.sum()
        cos = np.dot(phi, phi_norm) / (np.linalg.norm(phi) * np.linalg.norm(phi_norm))
        phis.append(1 - cos)

    print('Avg specialization level 2')
    print(sum(phis) / len(phis))

    # Topic specialization for level 3
    level_0 = community_topic.get_root_level_0()
    topics = community_topic.get_topic_hierarchical()
    level = topics[3]
    print("level 3 topics: ", len(level))
    phi_norm = np.array([community_topic.sentence_nb.token_freqs[v] for v in level_0])
    phi_norm = phi_norm / phi_norm.sum()
    phis = []

    for t in level.values():
        topic = t["dict_num"]
        phi = get_topic_phi(topic, community_topic.nx_g, level_0)
        phi = phi / phi.sum()
        cos = np.dot(phi, phi_norm) / (np.linalg.norm(phi) * np.linalg.norm(phi_norm))
        phis.append(1 - cos)

    print('Avg specialization level 3')
    print(sum(phis) / len(phis))

    # Topic Affinity for level 1  & 2
    child_sims = []
    nonchild_sims = []

    supertopic_level = topics[1]
    subtopic_level = topics[2]

    for i, supertopic in supertopic_level.items():
        i = int(i)
        supertopic = supertopic["dict_num"]
        super_phi = get_topic_phi(supertopic, community_topic.nx_g, level_0)

        for j, subtopic in subtopic_level.items():
            j = int(j.split("_")[0])
            subtopic = subtopic["dict_num"]
            sub_phi = get_topic_phi(subtopic, community_topic.nx_g, level_0)
            cos = np.dot(super_phi, sub_phi) / (np.linalg.norm(super_phi) * np.linalg.norm(sub_phi))
            if i == j:
                child_sims.append(cos)
            else:
                nonchild_sims.append(cos)

    avg_child_sims = sum(child_sims) / len(child_sims)
    avg_nonchild_sims = sum(nonchild_sims) / len(nonchild_sims)

    print("Hierarchical Affinity")
    print(avg_child_sims, "(child)")
    print(avg_nonchild_sims, "(non-child)")

    # ToDO: Topic affinity for level 3
    # Means what?
    # - level 3 affinity with level 2, 0_0 with all the other 0_0_* (child) and all other *_*_* (non child)
    # means I have to for all level 2 as my super topic, and level 3 as my subtopic
    # and i = 0_0 and j = 0_0_* (only 0_0 split from this)

    # or is it like level 3's with level 1's which is parent super topic?

    # which way would be better to do?

    print(f"community topic test on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
