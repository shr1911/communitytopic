# https://towardsdatascience.com/hands-on-topic-modeling-with-python-1e3466d406d7
# https://towardsdatascience.com/topic-model-visualization-using-pyldavis-fecd7c18fbf6

# Topic within topic graph like this - https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# Do different ideas in this link

# for Hierarchical - https://stackoverflow.com/questions/51903172/how-to-display-a-tree-in-python-similar-to-msdos-tree-command/51920869#51920869

#https://towardsdatascience.com/beyond-the-cloud-4-visualizations-to-use-instead-of-word-cloud-960dd516f215
from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time
import pickle


def main():
    """
    This is a visualization simple topic evaluation which contains the best combination for pre-processing and CommunityTopic Algorithm.

    1. Pre-processes training and testing corpus
    2. Apply CommunityTopic Algorithm
    3. Get topic words
    4. Visualize the topics
    """
    with open("../../text_datasets/europarl_en_train.txt", "r", encoding='utf-8') as f:
        bbc_train = f.read()
    with open("../../text_datasets/europarl_en_test.txt", "r", encoding='utf-8') as f:
        bbc_test = f.read()

    tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
        train=bbc_train,
        test=bbc_test,
        ner=1,
        pos_filter=3,
        phrases="npmi",
        phrase_threshold=0.35,
        language="en")

    with open("../tokenized_bbc_train_sents.obj", "wb") as f:
        pickle.dump(tokenized_bbc_train_sents, f)

    with open("../tokenized_bbc_train_docs.obj", "wb") as f:
        pickle.dump(tokenized_bbc_train_docs, f)

    with open("../tokenized_bbc_test_docs.obj", "wb") as f:
        pickle.dump(tokenized_bbc_test_docs, f)

    with open("../dictionary.obj", "wb") as f:
        pickle.dump(dictionary, f)

    t0 = time()
    community_topic = CommunityTopic(train_corpus=tokenized_bbc_train_sents,
                                     dictionary=dictionary,
                                     edge_weight="npmi",
                                     weight_threshold=0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1,
                                     network_window="sentence")

    community_topic.fit()
    t1 = time()

    topic_words = community_topic.get_topics_words()
    print(community_topic.get_topics_words_topn(10))
    print(f"Num topics: {len(topic_words)}")

    print(f"community topic test on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
