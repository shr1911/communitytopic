# https://stackoverflow.com/questions/66449183/wordcloud-and-list-of-list
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pickle


def main():
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

    community_topic.fit()
    t1 = time()

    topic_words = community_topic.get_topics_words()
    print(topic_words)
    print(f"Num topics: {len(topic_words)}")

    print(f"community topic on bbc finished in {t1 - t0} seconds")

    corpus = " ".join(" ".join(x) for x in topic_words)
    df_wordcloud = WordCloud(background_color='white', max_font_size=50).generate(corpus)
    plt.imshow(df_wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
