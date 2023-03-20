import pickle
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from time import time
import numpy as np
from diversity_metrics import *


def main():
    ner = "1"
    pos_filter = "0"
    phrase = "npmi"
    phrase_threshold = "0.35"

    with open("./ep2_master_object.obj", "rb") as f:
        master_object = pickle.load(f)

    # ng_dict = master_object["ng_dict"]
    # rt_dict = master_object["rt_dict"]
    # bbc_dict = master_object["bbc_dict"]
    ep_dict = master_object["ep_dict"]

    # f = open("bbc2_lda_results.csv", "a")

    # first lets evaluate LDA
    # let's use 20NG first
    # t0 = time()
    # for n_topics in [5, 10, 20, 50, 100, 200]:
    #     corpus = [ng_dict.doc2bow(text) for text in master_object["ng_train"]]
    #     t00 = time()
    #     lda = LdaModel(corpus, num_topics=n_topics, iterations=1000)
    #     t11 = time()
    #     print(f"LDA {n_topics} topics {t11 - t00} seconds")
    #     for ref in ["test"]:
    #         key = "ng_" + ref
    #         ref_corpus = master_object[key]
    #         for coherence in ["c_v", "c_npmi"]:
    #             for topn in [5, 10, 20]:
    #                 cm = CoherenceModel(model=lda, texts=ref_corpus, dictionary=ng_dict, topn=topn, coherence=coherence)
    #                 score = cm.get_coherence()
    #                 row = f"ng,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},lda,{n_topics},na,na,na,na,{coherence},{topn},{score}"
    #                 f.write(row + "\n")
    # t1 = time()
    # print(f"LDA on 20 NG finished in {t1 - t0} seconds")

    # now we'll do reuters
    # t0 = time()
    # for n_topics in [5, 10, 20, 50, 100, 200]:
    #     corpus = [rt_dict.doc2bow(text) for text in master_object["rt_train"]]
    #     t00 = time()
    #     lda = LdaModel(corpus, num_topics=n_topics, iterations=2000)
    #     t11 = time()
    #     print(f"LDA {n_topics} topics {t11 - t00} seconds")
    #     for ref in ["test"]:
    #         key = "rt_" + ref
    #         ref_corpus = master_object[key]
    #         for coherence in ["c_v", "c_npmi"]:
    #             for topn in [5, 10, 20]:
    #                 cm = CoherenceModel(model=lda, texts=ref_corpus, dictionary=rt_dict, topn=topn, coherence=coherence)
    #                 score = cm.get_coherence()
    #                 row = f"rt,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},lda,{n_topics},na,na,na,na,{coherence},{topn},{score}"
    #                 f.write(row + "\n")
    # t1 = time()
    # print(f"LDA on RT finished in {t1 - t0} seconds")

    t0 = time()
    for n_topics in [5]:
        corpus = [ep_dict.doc2bow(text) for text in master_object["ep_train"]]
        type(corpus)
        t00 = time()
        lda = LdaModel(corpus, num_topics=n_topics, iterations=2000)
        t11 = time()
        print(f"LDA {n_topics} topics {t11 - t00} seconds")

        topic_words = []
        # Need to generate list of list (words)
        for i in range(n_topics):
            topics = lda.get_topic_terms(i, 10)
            # print(topics)
            words = []
            for item in topics:
                words.append(ep_dict[item[0]])
                # print(item[0], " ", ng_dict[item[0]])
            # print(words)
            topic_words.append(words)
        print(topic_words)

        print("puw:", proportion_unique_words(topic_words, topk=10))
        print("jd:", pairwise_jaccard_diversity(topic_words, topk=10))
        print("irbo p=0.5:", irbo(topic_words, weight=0.5, topk=10))
        print("irbo p=0.9:", irbo(topic_words, weight=0.9, topk=10))

        for ref in ["test"]:
            key = "ep_" + ref
            ref_corpus = master_object[key]
            for coherence in ["c_v", "c_npmi", "u_mass"]:
                for topn in [5, 10, 20]:
                    cm = CoherenceModel(model=lda, texts=ref_corpus, dictionary=ep_dict, topn=topn,
                                        coherence=coherence)
                    score = cm.get_coherence()
                    print(coherence, ":", score, "(topn=", topn, ")")

    t1 = time()
    print(f"LDA finished in {t1 - t0} seconds")

    f.close()


if __name__ == "__main__":
    main()
