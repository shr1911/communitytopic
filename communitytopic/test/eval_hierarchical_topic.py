from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time
import pickle
import numpy as np
from typing import Union, Iterable, Iterator, List, Optional, Set, Dict, Tuple
import networkx as nx


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
    with open("./tokenized_bbc_train_sents.obj", "rb") as f:
        tokenized_bbc_train_sents = pickle.load(f)

    with open("./tokenized_bbc_train_docs.obj", "rb") as f:
        tokenized_bbc_train_docs = pickle.load(f)

    with open("./tokenized_bbc_test_docs.obj", "rb") as f:
        tokenized_bbc_test_docs = pickle.load(f)

    with open("./dictionary.obj", "rb") as f:
        dictionary = pickle.load(f)

    t0 = time()
    community_topic = CommunityTopic(corpus=tokenized_bbc_train_sents,
                                     dictionary=dictionary,
                                     edge_weight="npmi",
                                     weight_threshold=0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1,
                                     network_window="sentence")

    community_topic.fit_hierarchical(3)
    t1 = time()

    topic_words = community_topic.get_n_level_topic_words_hierarchical(3)

    eval_topics = []

    for key, level in topic_words.items():
        for i, topic in level.items():
            eval_topics.append(topic)

    # Coherence scores for hierarchical
    # for coherence in ["c_v", "c_npmi"]:
    #     cm = CoherenceModel(topics=eval_topics, texts=tokenized_bbc_test_docs, dictionary=dictionary, topn=5,
    #                         coherence=coherence)
    #     score = cm.get_coherence()
    #     print(f'{coherence}: {score}')

    # Topic Specialization for level 2
    level_0 = community_topic.get_root_level_0()
    topics = community_topic.get_topic_hierarchical()
    print(topics)
    print(len(topics))

    level = topics[2]

    phi_norm = np.array([community_topic.sentence_nb.token_freqs[v] for v in level_0])
    phi_norm = phi_norm / phi_norm.sum()

    phis = []

    for t in level.values():
        topic = t["dict_num"]
        phi = get_topic_phi(topic, community_topic.nx_g, level_0)
        phi = phi / phi.sum()
        cos = np.dot(phi, phi_norm) / (np.linalg.norm(phi) * np.linalg.norm(phi_norm))
        phis.append(1 - cos)

    print('Avg specialization level 2')
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

    print(avg_child_sims)
    print(avg_nonchild_sims)

    # Topic specialization for level 3
    level_0 = community_topic.get_root_level_0()
    topics = community_topic.get_topic_hierarchical()
    print(topics)
    print(len(topics))

    level = topics[3]

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
