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

import pickle


def main():
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

    community_topic.fit()
    t1 = time()

    topic_words = community_topic.get_topics()
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
