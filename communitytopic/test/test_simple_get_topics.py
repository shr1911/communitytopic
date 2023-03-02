from communitytopic import CommunityTopic
from communitytopic import PreProcessing
from gensim.models.coherencemodel import CoherenceModel
from time import time



def main():
    with open("../text_datasets/bbc_train.txt", "r") as f:
        bbc_train = f.read();
    with open("../text_datasets/bbc_test.txt", "r") as f:
        bbc_test = f.read();

    tokenized_train_corpus_docs, dictionary = PreProcessing.do_preprocessing(bbc_train, ner=1,
                                                                             pos_filter=0,
                                                                             phrases="npmi",
                                                                             phrase_threshold=0.35)

    tokenized_test_corpus_docs, test_dictionary = PreProcessing.do_preprocessing(bbc_test, ner=1,
                                                                                 pos_filter=0,
                                                                                 phrases="npmi",
                                                                                 phrase_threshold=0.35)

    t0 = time()
    community_topic = CommunityTopic(corpus=tokenized_train_corpus_docs,
                                     dictionary=dictionary,
                                     edge_weight="count",
                                     weight_threshold=0.0,
                                     cd_algorithm="leiden",
                                     resolution_parameter=1.0,
                                     network_window="sentence")

    community_topic.fit()

    topic_words = community_topic.get_topics()
    print(topic_words)
    print(f"Num topics: {len(topic_words)}")

    cm = CoherenceModel(topics=topic_words,
                        texts=tokenized_test_corpus_docs,
                        dictionary=dictionary,
                        topn=5,
                        coherence="c_v")
    score = cm.get_coherence()
    print(score)
    t1 = time()
    print(f"community topic test on bbc finished in {t1 - t0} seconds")

if __name__ == '__main__':
    main()
