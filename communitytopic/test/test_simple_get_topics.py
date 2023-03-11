from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time
import pickle

# This file contains perfect combinations for pre-processing and CT algorithm

def main():
    with open("../text_datasets/bbc_train.txt", "r") as f:
        bbc_train = f.read();
    with open("../text_datasets/bbc_test.txt", "r") as f:
        bbc_test = f.read();

    tokenized_bbc_train_sents, tokenized_bbc_train_docs, tokenized_bbc_test_docs, dictionary = PreProcessing.do_preprocessing(
        bbc_train, bbc_test, ner=1,
        pos_filter=3,
        phrases="npmi",
        phrase_threshold=0.35)

    with open("./tokenized_bbc_train_sents.obj", "wb") as f:
        pickle.dump(tokenized_bbc_train_sents, f)

    with open("./tokenized_bbc_train_docs.obj", "wb") as f:
        pickle.dump(tokenized_bbc_train_docs, f)

    with open("./tokenized_bbc_test_docs.obj", "wb") as f:
        pickle.dump(tokenized_bbc_test_docs, f)

    with open("./dictionary.obj", "wb") as f:
        pickle.dump(dictionary, f)

    t0 = time()
    community_topic = CommunityTopic(corpus=tokenized_bbc_train_sents,
                                     dictionary=dictionary,
                                     edge_weight="npmi",
                                     weight_threshold=0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1,
                                     network_window="sentence")

    community_topic.fit()

    topic_words = community_topic.get_topics()
    print(topic_words)
    print(f"Num topics: {len(topic_words)}")

    cm = CoherenceModel(topics=topic_words,
                        texts=tokenized_bbc_test_docs,
                        dictionary=dictionary,
                        topn=5,
                        coherence="c_v")
    score = cm.get_coherence()
    print(score)
    t1 = time()
    print(f"community topic test on bbc finished in {t1 - t0} seconds")


if __name__ == '__main__':
    main()
