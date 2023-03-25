import collections
import logging
import networkx as nx
import igraph as ig
from collections import defaultdict
from typing import Union, Iterable, Iterator, List, Optional, Set, Dict, Tuple
import math
from gensim.corpora import Dictionary
import time

logger = logging.getLogger('communitytopic')


def get_internal_weighted_degree(index: int, community: List[str], graph: nx.Graph):
    """
    Given a term index and a graph, return the internal weighted degree of node corresponding to term,
    i.e. the sum of edge strengths connecting to nodes in same community
    """
    community_subgraph = graph.subgraph(community)
    return community_subgraph.degree(weight="weight")[str(index)]


class CommunityTopic:
    """ Community Topic, that mines communities from word co-occurrence networks to produce topics.

    Simple Example:
    --------------
    Step 1: Make necessary imports
        from communitytopic import CommunityTopic
        from communitytopic import PreProcessing


    Step 2: Do pre-processing (Best pre-processing parameters are used in this example).
        tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
                train=bbc_train,
                test=bbc_test,
                ner=1,
                pos_filter=3,
                phrases="npmi",
                phrase_threshold=0.35,
                language="en")


    Step 3: Initialize CommunityTopic
        community_topic = CommunityTopic(corpus=tokenized_bbc_train_sents,
                                        dictionary=dictionary,
                                        edge_weight="npmi",
                                        weight_threshold=0,
                                        cd_algorithm="leiden",
                                        resolution_parameter=1,
                                        network_window="sentence")

    Step 4: Fitting and applying CommunityTopic to find out methods
        community_topic.fit()

    Step 5: Getting topic words
        topic_words = community_topic.get_topics_words()
        print(topic_words)
        print(f"Num topics: {len(topic_words)}")
    """

    def __init__(self, train_corpus=None, dictionary=None, edge_weight="count", weight_threshold=0.0,
                 cd_algorithm="leiden", resolution_parameter=1.0, network_window="sentence"):
        """

        @param train_corpus: List of List of (String words)
            Preprocessed sentences of training corpus (List of list)
            It contains pre-processed tokenized sentence as list of list

        @param dictionary: dict
            Gensim dictionary object that tracks frequencies and can filter vocab
            - keys are id for words
            - values are words

        @param edge_weight: String
            It is weight of edges which comes from the frequency of co-occurrence.
            Possible values: ["count", "npmi"]
            "count":  Raw count of possible edges as the edge weight.
            "npmi": Weighing scheme which uses Normalized Pointwise Mutual Information (NPMI) between terms

        @param weight_threshold: float
            The edges can be thresholded, i.e. those edges whose weights fall below a
            certain threshold are removed from the network.

        @param cd_algorithm: String
            To choose community detection algorithm
            Possible values: ["leiden", "walktrap"]

        @param resolution_parameter: float
            Te resolution_parameter to use for leiden community detection algorithm.
            Higher resolution_parameter lead to smaller communities, while
            lower resolution_parameter lead to fewer larger communities.

        @param network_window:
            The network that we construct from a corpus has terms as vertices. This decides the
                fixed sliding window of document.
            Possible values: ["sentence", "5", "10"]
            "sentence": two terms co-occur if they both occur in the same sentence.
            "5" or "10": two terms co-occur if they both occur within a fixed-size sliding window over a document.


        """
        # Input parameters
        self.train_corpus = train_corpus
        self.edge_weight = edge_weight
        self.weight_threshold = weight_threshold
        self.cd_algorithm = cd_algorithm
        self.resolution_parameter = resolution_parameter
        self.network_window = network_window
        self.dictionary = dictionary


        # Public attributes
        self.topics = None
        self.topics_words = None
        self.hierarchical_topics = collections.defaultdict(dict)
        self.hierarchical_topics_words = collections.defaultdict(dict)
        self.level_0 = None

        # Private attributes for internal tracking purposes
        self.master_object = dict()
        self.nx_g = None
        self.ig_g = None
        self.clustering = None
        self.level_1 = collections.defaultdict(dict)
        self.sentence_nb = None

    def fit(self):
        """
        This method performs task of finding simple topics
        1. Network generations
        2. Finding communities
        3. Prune communities greater than 2
        4. Convert it into topic words using vocab dictionary

        """

        # Generates network as per given network_window parameter
        print("Generating network...")
        t0 = time.time()
        if self.network_window == "sentence":
            self.sentence_nb = SentenceNetworkBuilder(self.train_corpus, self.dictionary)
        else:
            self.sentence_nb = WindowNetworkBuilder(self.train_corpus, self.dictionary,
                                                    int(self.network_window))
        self.sentence_nb.save_network(f"./network.txt", edge_weight=self.edge_weight, threshold=self.weight_threshold)

        # Converts the graph from networkx
        # Vertex names will be converted to "_nx_name" attribute and the vertices will get new ids from 0 up.
        # Refer for more on igraph api: https://igraph.org/python/doc/api/igraph.Graph.html
        self.nx_g = nx.read_weighted_edgelist("./network.txt")
        self.ig_g = ig.Graph.from_networkx(self.nx_g)
        t1 = time.time()

        # Finds the community structure of the graph using the Leiden algorithm of Traag, van Eck & Waltman.
        print("Finding topic communities...")
        t0 = time.time()
        if self.cd_algorithm == 'leiden':
            self.clustering = self.ig_g.community_leiden(resolution=self.resolution_parameter,
                                                         weights='weight', objective_function='modularity')
        else:
            self.clustering = self.ig_g.community_walktrap(weights='weight').as_clustering()
        t1 = time.time()
        print(f"Topics found in {t1 - t0} seconds")

        # Storing all the root level words into level_0 (for later use for hierarchical topic detection)
        self.level_0 = [int(self.ig_g.vs[v.index]["_nx_name"]) for v in self.ig_g.vs]

        print("Sorting topics...")
        t0 = time.time()
        # Pruning communities which are lesser than length 2
        comms = [[int(self.ig_g.vs[node]["_nx_name"]) for node in comm] for comm in self.clustering if len(comm) > 2]

        # Rank each words in the topic using internal_weighted_degree
        # Also storing dictionary-id as topic into level_1
        i = 0
        for comm in comms:
            c = [str(node) for node in comm]
            comm.sort(key=lambda node: get_internal_weighted_degree(node, c, self.nx_g), reverse=True)
            self.level_1[str(i)]["dict_num"] = comm
            i = i + 1

        # Use dictionary to get topic words from dictionary ids
        self.topics = comms
        self.topics_words = [[self.dictionary[node] for node in comm] for comm in comms]
        t1 = time.time()
        print(f"Topics sorted in {t1 - t0} seconds")

        # Storing igraph of each level 1 topics for hierarchical topics
        i = 0
        for cluster in self.clustering:
            self.level_1[str(i)]["ig_graph"] = self.ig_g.subgraph(cluster)
            i = i + 1

    def fit_hierarchical(self, n_level=2):
        """
        This method performs task of finding hierarchical topics
        1. Calling fit() method first, which finds level_1/simple topics
        2. Generating next level using information stored in level_1
        3. Keep generating next levels for n_level number of times
        4. At this point hierarchical topics are generated
        5. Next, convert it into topic words using vocab dictionary

        @param n_level:int
            Number of level for hierarchical topics

        """
        self.hierarchical_topics.clear()

        # Generate first level of hierarchical topics
        self.fit()

        # Generate next remaining levels in hierarchy
        next_level = self.level_1
        self.hierarchical_topics[1] = self.level_1
        for i in range(2, n_level + 1):
            next_level = self.next_level_generation(next_level)
            # clean the generated level: remove empty sub-topics entries from the generated level
            next_level = self.do_level_cleaning(next_level)
            self.hierarchical_topics[i] = next_level

        # Use dictionary to get topic words from dictionary ids (for each levels)
        for n, level in self.hierarchical_topics.items():
            level_topic_words = dict()
            for key, topic in level.items():
                try:
                    level_topic_words[key] = [self.dictionary[node] for node in topic["dict_num"]]
                except:
                    print("issue: ", key, topic)
            self.hierarchical_topics_words[n] = level_topic_words

    def next_level_generation(self, level):
        """
        Given the parent topic level, generate next child level of topics in hierarchy
        @param level: dict of dict
            Dictionary of each topic in that level
            Each topic's value is dictionary of 'dict_num' and 'ig_graph'.
                'dict_num' - dictionary id of corresponding words
                'ig_graph' - ig_graph object that topic

        @return:
            next_level: dict of dict
        """
        next_level = collections.defaultdict(dict)

        for key, value in level.items():
            j = 0
            k = 0
            subclusters = value["ig_graph"].community_leiden(objective_function='modularity', resolution=1.0,
                                                             weights='weight')
            # print(f'{len(subclusters)} sub clusters found')
            stopics_dict_num = [[int(value["ig_graph"].vs[node]["_nx_name"]) for node in comm] for comm in subclusters
                                if len(comm) > 2]
            for comm in stopics_dict_num:
                c = [str(node) for node in comm]
                comm.sort(key=lambda node: get_internal_weighted_degree(node, c, self.nx_g), reverse=True)
                # print(str(key) + "_" + str(j))
                next_level[str(key) + "_" + str(j)]["dict_num"] = comm
                j = j + 1
            for cluster in subclusters:
                next_level[str(key) + "_" + str(k)]["ig_graph"] = value["ig_graph"].subgraph(cluster)
                k = k + 1
            j = j + 1
            k = k + 1

        # print(next_level)
        return next_level

    def do_level_cleaning(self, level):
        cleaned_level = collections.defaultdict(dict)

        for key, topic in level.items():
            if len(topic) == 2:
                # print(key, len(topic), topic)
                cleaned_level[key] = topic
        return cleaned_level

    def get_topics_words(self):
        """
        Get topic words

        @return:
            topics: list of list
            Returns simple topics as topic words
        """
        return self.topics_words

    def get_topics(self):
        """
        Get topic as dictionary id

        @return:
            topics: list of list
            Returns simple topics as dictionary id
        """
        print("get_topics")
        return self.topics

    def get_topics_topn(self, n=10):
        """
        Get top n topic words as dictionary id for each topic

        @param
            n: int
            top n topic words as dictionary id

        @return:
            topics: list of list
            Returns simple topics as dictionary id
        """
        topn_topics = []
        for topic in self.topics:
            topn_topics.append(topic[:n])
        return topn_topics

    def get_topics_words_topn(self, n=10):
        """
        Get top n topic words for each topic

        @param
            n: int
            top n topic words

        @return:
            topics: list of list
            Returns topics words as list of list

        """
        topn_topic_words = []
        for topic in self.topics_words:
            topn_topic_words.append(topic[:n])
        return topn_topic_words

    def get_topic_words_hierarchical(self):
        """
        Get hierarchical topic as topic words

        @return:
            hierarchical_topics_words: dict of dict

            In following format (each level and topic in that level)-
            {1 : {"0": ['firm', 'company', 'economy',...],
                  "1": ['country', 'china', 'bank'....], }
                  .....},
            2 : {"0": [''orders', 'spring', 'allies',...],
                 "1": ['lawyer', 'individuals', 'failure'....], }
                  .....},
            .....
            }
        """
        return self.hierarchical_topics_words

    def get_topics_hierarchical(self):
        """
        Get hierarchical topic as dictionary id, and ig_graph of topic

        @return:
            hierarchical_topics: dict of dict

            In following format (each level and topic in that level)-
            {1 : {"0":  {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                  "1": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                  .....},
            2 : {"0_0": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                 "0_1": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                 .....
                 ....
                 "1_0": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                 "1_1": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                  .....},
            3 : {"0_0_0": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                "0_0_1": {'dict_num': [2, 147, 6, 1180, 327, ,....]
                         'ig_graph': object of ig_graph
                        },
                .....
                },
            ......
            }

            Note, in above format each level has topic names as the key of dictionary.
                For example, level 1 has single digit value which specifies topics in that level
                level 2 has two values seperated by underscore, first value is super topic and second is child topic
                Similary, level 3 has three values, for which parent topics and current child topic
        """
        return self.hierarchical_topics

    def get_n_level_topic_words_hierarchical(self, n_level=2):
        """
        Get first n number of levels from hierarchy

        @param
            n_level: int

        @return:
            topics: dict of dict
        """
        topic_words = collections.defaultdict(dict)

        for i in range(1, n_level + 1):
            topic_words[i] = self.hierarchical_topics_words[i]

        return topic_words

    def get_num_levels_count(self):
        """
        Returning numer of levels in hierarchy

        @return: int
        Count of levels
        """
        return len(self.hierarchical_topics_words)

    def get_nth_level(self, n=1):
        hierarchical_n_topics_words = collections.defaultdict()
        for key, topic in self.hierarchical_topics_words[n].items():
            hierarchical_n_topics_words[key] = topic[:10]
        return hierarchical_n_topics_words

    def get_root_level_0(self):
        """
        Get root level words of corpus

        @return:
            level_0: list
            Root words of  corpus as level_0
        """
        return self.level_0

    def get_hierarchy_tree(self):
        """
        This function is for visualisation purpose of hierarchical topics.

        It returns a tree-like structure in dictionary format.

        """
        input_data = {}

        for i in range(1, self.get_num_levels_count() + 1):
            dictionary = self.get_nth_level(i)
            for key in dictionary:
                input_data[key] = ' '.join(dictionary[key])

        tree = self.construct_tree(input_data)

        return tree

    def construct_tree(self, input_data):
        """
        Constructs a tree structure from the given dictionary data.
        """
        tree = defaultdict(dict)
        for key, value in input_data.items():
            path = key.split('_')
            self.add_node(tree, path, value)
        return tree

    def add_node(self, tree, path, value):
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


class SentenceNetworkBuilder:
    """
    Class for constructing a word co-occurrence network from a series of tokenized documents for given sentence.
    """

    def __init__(
            self,
            sentences: Iterable[Iterable[str]],
            dictionary: Dictionary
    ) -> None:
        """

        @param sentences: list of list
            pre-processed training corpus

        @param dictionary: dict
            Gensim dictionary object that tracks frequencies and can filter vocab
            - keys are id for words
            - values are words

        """
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
            edge_weight: str = "default",
            threshold: Union[int, float] = 0
    ) -> None:
        """
        To save sentence based network to given file location

        @param filepath: String
            Path for the file, where generated network will be stored

        @param edge_weight: String
            It is weight of edges which comes from the frequency of co-occurrence.
            Possible values: ["count", "npmi"]
            "count":  Raw count of possible edges as the edge weight.
            "npmi": Weighing scheme which uses Normalized Pointwise Mutual Information (NPMI) between terms

        @param threshold: float
            The edges can be thresholded, i.e. those edges whose weights fall below a
            certain threshold are removed from the network.
        """
        # save weighted edgelist
        lines = []
        for edge, count in self.edge_counts.items():
            if edge_weight == "npmi":
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
    Class for constructing a word co-occurrence network from a series of tokenized documents for given window size.
    """

    def __init__(
            self,
            docs: Iterable[Iterable[str]],
            dictionary: Dictionary,
            window: int = 5
    ) -> None:
        """

        @param docs: list of list
            pre-processed training corpus

        @param dictionary: dict
            Gensim dictionary object that tracks frequencies and can filter vocab
            - keys are id for words
            - values are words

        @param window: int
            Size of fixed-size window
        """

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
            edge_weight: str = "default",
            threshold: Union[int, float] = 0
    ) -> None:
        """
        To save window based network to given file location

        @param filepath: String
            Path for the file, where generated network will be stored

        @param edge_weight: String
            It is weight of edges which comes from the frequency of co-occurrence.
            Possible values: ["count", "npmi"]
            "count":  Raw count of possible edges as the edge weight.
            "npmi": Weighing scheme which uses Normalized Pointwise Mutual Information (NPMI) between terms

        @param threshold: float
            The edges can be thresholded, i.e. those edges whose weights fall below a
            certain threshold are removed from the network.
        """
        # save weighted edgelist
        lines = []
        for edge, count in self.edge_counts.items():
            if edge_weight == "npmi":
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
