import logging
import networkx as nx
import igraph as ig
from collections import defaultdict
from typing import Union, Iterable, Iterator, List, Optional, Set, Dict, Tuple
import math
from gensim.corpora import Dictionary
from time import time

logger = logging.getLogger('communitytopic')


def get_term_frequency(index: int, dictionary: Dictionary):
    """
    Given a term index and a dictionary, look up the frequency of that term
    """
    return dictionary.cfs[index]


def get_degree(index: int, graph: nx.Graph):
    """
    Given a term index and graph, return the degree of the node corresponding to term
    """
    return graph.degree[str(index)]


def get_weighted_degree(index: int, graph: nx.Graph):
    """
    Given a term index and graph, return the weighted degree of the node corresponding to term
    """
    return graph.degree(weight='weight')[str(index)]


def get_internal_degree(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the internal degree of node corresponding to term,
    i.e. the number of edges connecting to nodes in same community
    """
    community_subgraph = graph.subgraph(community)
    return community_subgraph.degree[str(index)]


def get_internal_weighted_degree(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the internal weighted degree of node corresponding to term,
    i.e. the sum of edge strengths connecting to nodes in same community
    """
    community_subgraph = graph.subgraph(community)
    return community_subgraph.degree(weight="weight")[str(index)]


def get_weighted_embeddedness(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the weighted embeddedness of node corresponding to term,
    i.e. ratio of internal weighted degree to total weighted degree
    """
    internal_degree = get_internal_weighted_degree(index, community, graph)
    degree = get_weighted_degree(index, graph)
    return internal_degree / degree


def get_embeddedness(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the embeddedness of node corresponding to term,
    i.e. ratio of internal degree to total degree
    """
    internal_degree = get_internal_degree(index, community, graph)
    degree = get_degree(index, graph)
    return internal_degree / degree


class CommunityTopic:
    """

    """

    def __init__(self, corpus=None, dictionary=None, edge_weight="count", weight_threshold=0.0,
                 cd_algorithm="leiden", resolution_parameter=1.0, network_window="sentence"):

        # Input parameters
        self.corpus = corpus
        self.dictionary = dictionary
        self.edge_weight = edge_weight
        self.weight_threshold = weight_threshold
        self.cd_algorithm = cd_algorithm
        self.resolution_parameter = resolution_parameter
        self.network_window = network_window

        # Public attributes
        self.topics = None

        # Private attributes for internal tracking purposes
        self.master_object = dict()
        self.nx_g = None
        self.ig_g = None
        self.clustering = None
        self.level_0 = None

    def fit(self):

        # Do all pre-processing, network generations, saving network,
        # finding communities, prune commmunities greater than 2
        # keep ready topics as terms in variable

        print("Generating network...")
        t0 = time()
        if self.network_window == "sentence":
            sentence_nb = SentenceNetworkBuilder(self.corpus, self.dictionary)
        else:
            sentence_nb = WindowNetworkBuilder(self.corpus, self.dictionary,
                                               int(self.network_window))
        sentence_nb.save_network(f"./network.txt", type=self.edge_weight, threshold=self.weight_threshold)
        self.nx_g = nx.read_weighted_edgelist("./network.txt")
        self.ig_g = ig.Graph.from_networkx(self.nx_g)
        t1 = time()
        print("fit")

        print("Finding topic communities...")
        t0 = time()
        if self.cd_algorithm == 'leiden':
            self.clustering = self.ig_g.community_leiden(resolution=self.resolution_parameter,
                                          weights='weight', objective_function='modularity')
        else:
            self.clustering = self.ig_g.community_walktrap(weights='weight').as_clustering()
        t1 = time()
        print(f"Topics found in {t1 - t0} seconds")

        self.level_0 = [int(self.ig_g.vs[v.index]["_nx_name"]) for v in self.ig_g.vs]

        print("Sorting topics...")
        t0 = time()
        comms = [[int(self.ig_g.vs[node]["_nx_name"]) for node in comm] for comm in self.clustering if len(comm) > 2]
        for comm in comms:
            c = [str(node) for node in comm]
            comm.sort(key=lambda node: get_internal_weighted_degree(node, c, self.nx_g), reverse=True)
        self.topics = [[self.dictionary[node] for node in comm] for comm in comms]
        t1 = time()
        print(f"Topics sorted in {t1 - t0} seconds")

    def get_topics(self):
        # return topics as list of list
        print("get_topics")
        return self.topics

    def get_topics_topn(self, n):
        # return topics as list of list
        print("get_topics_topn")
        topn_topics = []
        for topic in self.topics:
            topn_topics.append(topic[:n])
            print(topic[:n])

    def visualize_topics(self):
        # refer bertopic for this
        print("visualize_topics")

    def fit_hierarchical(self):
        print("fit_hierarchical")

    def hierarchical_topic(self, n_level=2):
        print("hierarchical_topic")
        self.fit()

        # Level 0 - root words
        # level 1 - first generalized topic (fit method)
        hierarchical_clustering = dict()
        hierarchical_topic = dict()
        hierarchical_ig = dict()

        hierarchical_topic[0] = self.level_0

        hierarchical_topic[1] = self.topics
        hierarchical_clustering[1] = self.clustering
        hierarchical_ig[1] = self.ig_g


        for i in range(2, n_level+1):
            subclusterings = []
            subtopics = []
            for cluster in hierarchical_clustering[i-1]:
                hierarchical_ig[i] = hierarchical_ig[i-1].subgraph(cluster)
                subclusters = hierarchical_ig[i].community_leiden(objective_function='modularity', resolution_parameter=1.0,
                                                 weights='weight')
                print(f'{len(subclusters)} sub clusters found')
                subclusterings.append(subclusters)
                stopics = [[int(hierarchical_ig[i].vs[node]["_nx_name"]) for node in comm] for comm in subclusters if len(comm) > 2]
                for comm in stopics:
                    c = [str(node) for node in comm]
                    comm.sort(key=lambda node: get_internal_weighted_degree(node, c, self.nx_g), reverse=True)

                stopics = [[self.dictionary[node] for node in comm] for comm in stopics]
                subtopics.append(stopics)
            hierarchical_clustering[i] = subclusterings
            hierarchical_topic[i] = subtopics
        return hierarchical_topic

    def get_hierarchical_topics(self):
        print("get_hierarchical_topics")


class SentenceNetworkBuilder:
    """
    Class for constructing a word co-occurrence network from a series of tokenized documents.
    """

    def __init__(
            self,
            sentences: Iterable[Iterable[str]],
            dictionary: Dictionary
    ) -> None:
        # initialize default dicts for counting token frequencies and edge counts
        self.total_tokens = 0
        self.token_freqs = defaultdict(int)
        self.edge_counts = defaultdict(int)

        for sentence in sentences:
            # eliminate duplicates
            sentence = list(set(sentence))
            # increase token counts
            for token in sentence:
                self.total_tokens += 1
                self.token_freqs[dictionary.token2id[token]] += 1
            # count co-occurrences
            for i in range(len(sentence) - 1):
                for j in range(i + 1, len(sentence)):
                    # create ordered edge (direction doesn't matter)
                    u = dictionary.token2id[sentence[i]]
                    v = dictionary.token2id[sentence[j]]
                    edge = (min(u, v), max(u, v))
                    self.edge_counts[edge] += 1

    def save_network(
            self,
            filepath: str,
            type: str = "default",
            threshold: Union[int, float] = 0
    ) -> None:
        # save weighted edgelist
        lines = []
        for edge, count in self.edge_counts.items():
            if type == "npmi":
                prob_u = self.token_freqs[edge[0]] / self.total_tokens
                prob_v = self.token_freqs[edge[1]] / self.total_tokens
                prob_uv = count / self.total_tokens
                try:
                    count = math.log(prob_uv / (prob_u * prob_v)) / -math.log(prob_uv)
                except ZeroDivisionError:
                    count = -1
            if count > threshold:
                lines.append(str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(count))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))


class WindowNetworkBuilder:
    """
    Class for constructing a word co-occurrence network from a series of tokenized documents.
    """

    def __init__(
            self,
            docs: Iterable[Iterable[str]],
            dictionary: Dictionary,
            window: int = 5
    ) -> None:
        # initialize default dicts for counting token frequencies and edge counts
        self.total_tokens = 0
        self.token_freqs = defaultdict(int)
        self.edge_counts = defaultdict(int)

        for doc in docs:
            for i in range(len(doc)):
                # get window
                window_tokens = doc[i:min(i + window, len(doc))]
                # only count first token
                u = dictionary.token2id[window_tokens[0]]
                self.total_tokens += 1
                self.token_freqs[u] += 1
                # remove duplicates and count co-occurrence
                vs = list(set(window_tokens[1:]))
                for token in vs:
                    v = dictionary.token2id[token]
                    if u != v:
                        edge = (min(u, v), max(u, v))
                        self.edge_counts[edge] += 1

    def save_network(
            self,
            filepath: str,
            type: str = "default",
            threshold: Union[int, float] = 0
    ) -> None:
        # save weighted edgelist
        lines = []
        for edge, count in self.edge_counts.items():
            if type == "npmi":
                prob_u = self.token_freqs[edge[0]] / self.total_tokens
                prob_v = self.token_freqs[edge[1]] / self.total_tokens
                prob_uv = count / self.total_tokens
                try:
                    count = math.log(prob_uv / (prob_u * prob_v)) / -math.log(prob_uv)
                except ZeroDivisionError:
                    count = -1
            if count > threshold:
                lines.append(str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(count))

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
