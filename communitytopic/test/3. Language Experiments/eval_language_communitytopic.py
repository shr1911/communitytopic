from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time
import pickle
from diversity_metrics import *


def main():
    """
    FOR DIFFERENT LANGUAGES
    (same files from simple topic experiments, just need to use different language dataset.
    Also, make sure to use appropriate pre-processing for that language)

    This is a simple topic evaluation which contains the best combination for pre-processing and CommunityTopic Algorith.

    1. Pre-processes on different language's training and testing corpus
    2. Apply CommunityTopic Algorithm (If using pre-processing of community-topic, put language option)
        "en" - English
        "it" - Italian
        "fr" - French
        "de" - German
        "es" - Spanish
    3. Get topic words
    4. Calculate coherence score ('c_v', 'c_npmi', 'c_umass') for testing data

    """
    with open("../../../text_datasets/europarl_de_train.txt", "r", encoding='utf-8') as f:
        bbc_train = f.read()
    with open("../../../text_datasets/europarl_de_test.txt", "r", encoding='utf-8') as f:
        bbc_test = f.read()

    tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
        train=bbc_train,
        test=bbc_test,
        ner=1,
        pos_filter=3,
        phrases="npmi",
        phrase_threshold=0.35,
        language="de")

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
                                     weight_threshold=0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1,
                                     network_window="sentence")

    community_topic.fit()
    t1 = time()

    topic_words = community_topic.get_topics_words()
    print(community_topic.get_topics_words_topn(10))
    print(f"Num topics: {len(topic_words)}")

    print("puw:", proportion_unique_words(topic_words, topk=10))
    print("jd:", pairwise_jaccard_diversity(topic_words, topk=10))
    print("irbo p=0.5:", irbo(topic_words, weight=0.5, topk=10))
    print("irbo p=0.9:", irbo(topic_words, weight=0.9, topk=10))

    # cm = CoherenceModel(topics=topic_words,
    #                     texts=tokenized_bbc_test_docs,
    #                     dictionary=dictionary,
    #                     topn=5,
    #                     coherence="c_v")
    # score = cm.get_coherence()
    # print(score)
    # t1 = time()

    for coherence in ["c_v", "c_npmi", "u_mass"]:
        for topn in [5, 10, 20]:
            cm = CoherenceModel(topics=topic_words,
                                texts=tokenized_bbc_test_docs,
                                dictionary=dictionary,
                                topn=topn,
                                coherence=coherence)
            score = cm.get_coherence()
            print(coherence, ":", score, "(topn=", topn, ")")
    print(f"community topic test on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
