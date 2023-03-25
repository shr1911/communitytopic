from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pickle


import matplotlib.pyplot as plt
import numpy as np

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

    topic_words = community_topic.get_topics_words_topn(n=10)
    print(topic_words)
    print(f"Num topics: {len(topic_words)}")

    # # Data for the chart
    # ranks = [5, 4, 3, 2, 1]
    #
    # # Topic 1
    # # Create a horizontal bar chart
    # plt.barh(topic_words[0], ranks, color='orange')
    #
    # # Add title and labels
    # plt.title('Topic 1', fontsize=14)
    # plt.xlabel('Rank', fontsize=14)
    # plt.ylabel('Word', fontsize=14)
    #
    # # Show the plot
    # plt.show()

    # import matplotlib.pyplot as plt
    #
    # # Create a figure and axis object
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # Plot each word list as a horizontal bar chart
    # for i, word_list in enumerate(topic_words):
    #     ax.barh(word_list, [10 - j for j in range(10)], height=0.8, color=f'C{i}', alpha=0.8, label=f'Topic {i + 1}')
    #
    # # Set axis labels and title
    # ax.set_xlabel('Rank')
    # ax.set_ylabel('Words')
    # ax.set_title('Top 10 ranked words topic-wise')
    #
    # # Add legend
    # ax.legend()
    #
    # # Show the plot
    # plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a matrix of ranks
    ranks = np.array([range(10, 0, -1), range(10, 0, -1), range(10, 0, -1), range(10, 0, -1), range(10, 0, -1)])

    # Create a heatmap
    sns.heatmap(ranks, cmap='YlGnBu', annot=np.array(topic_words), fmt='', annot_kws={"fontsize":15})

    # Set axis labels and title
    plt.xlabel('Rank',  fontsize=14)
    plt.ylabel('Topic', fontsize=14)
    plt.title('Top 10 ranked words topic wise', fontsize=14)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
