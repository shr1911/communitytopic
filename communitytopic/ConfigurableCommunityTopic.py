import logging

logger = logging.getLogger('communitytopic')


class ConfigurableCommunityTopic:
    """

    """

    def __init__(self, corpus=None, ner=1, pos_filter=0, phrases="npmi", phrase_threshold=0.35,
                 edge_weight="count", weight_threshold=0.0, cd_algorithm="leiden", resolution_parameter=1.0):
        self.corpus = corpus
        self.ner = ner
        self.pos_filter = pos_filter
        self.phrases = phrases
        self.phrase_threshold = phrase_threshold
        self.edge_weight = edge_weight
        self.weight_threshold = weight_threshold
        self.cd_algorithm = cd_algorithm
        self.resolution_parameter = resolution_parameter


