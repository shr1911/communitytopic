# https://towardsdatascience.com/hands-on-topic-modeling-with-python-1e3466d406d7
# https://towardsdatascience.com/topic-model-visualization-using-pyldavis-fecd7c18fbf6

# Topic within topic graph like this - https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# Do different ideas in this link

# for Hierarchical - https://stackoverflow.com/questions/51903172/how-to-display-a-tree-in-python-similar-to-msdos-tree-command/51920869#51920869

#https://towardsdatascience.com/beyond-the-cloud-4-visualizations-to-use-instead-of-word-cloud-960dd516f215
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

    print(f"community topic on bbc finished in {t1 - t0} seconds")

    topic_words = community_topic.get_topics_words()
    print(topic_words)
    print(f"Num topics: {len(topic_words)}")

    topic_1 = " ".join(topic_words[0])
    topic_2 = " ".join(topic_words[1])
    topic_3 = " ".join(topic_words[2])
    topic_4 = " ".join(topic_words[3])
    topic_5 = " ".join(topic_words[4])

    # corpus = " ".join(" ".join(x) for x in topic_words)
    # wc_t1 = WordCloud(background_color='white', max_font_size=50).generate(topic_1)
    # wc_t2 = WordCloud(background_color='white', max_font_size=50).generate(topic_2)
    # wc_t3 = WordCloud(background_color='white', max_font_size=50).generate(topic_3)
    # wc_t4 = WordCloud(background_color='white', max_font_size=50).generate(topic_4)
    # wc_t5 = WordCloud(background_color='white', max_font_size=50).generate(topic_5)
    # plt.imshow(df_wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()

    # Define a list of 5 colors for the word clouds
    color_list = ["#FF0000", "#008000", "#FFA500", "#0000FF", "#800080"]

    # Create a 2x3 subplot figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))

    text_list = [topic_1, topic_2, topic_3, topic_4, topic_5]

    # Loop through the text data and create a word cloud for each
    for i, ax in enumerate(axes.flatten()):
        if i < len(text_list):
            # Create the word cloud
            wc = WordCloud(background_color="white", max_words=50, width=800, height=400).generate(text_list[i])
            # Set the axis properties and display the word cloud
            ax.imshow(wc.recolor(color_func=lambda *args, **kwargs: color_list[i]), interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Topic {i + 1}")

    # Show the figure
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
