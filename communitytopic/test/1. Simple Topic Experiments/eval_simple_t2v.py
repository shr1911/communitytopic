from top2vec import Top2Vec
import pickle
from time import time
from gensim.models.coherencemodel import CoherenceModel
import tomotopy as tp
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



    # f = open("bbc2_t2v_results.csv", "a")

    # let's use 20NG first
    # t0 = time()
    # corpus = [" ".join(text) for text in master_object["ng_train"]]
    # t2v = Top2Vec(corpus, speed="learn")
    # topic_words, _, _ = t2v.get_topics()
    # for ref in ["test"]:
    #     key = "ng_" + ref
    #     ref_corpus = master_object[key]
    #     for coherence in ["c_v", "c_npmi"]:
    #         for topn in [5, 10, 20]:
    #             cm = CoherenceModel(topics=topic_words, texts=ref_corpus, dictionary=ng_dict, topn=topn, coherence=coherence)
    #             score = cm.get_coherence()
    #             row = f"ng,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},t2v,na,na,na,na,na,{coherence},{topn},{score}"
    #             f.write(row + "\n")
    # t1 = time()
    # print(f"t2v on 20 NG finished in {t1 - t0} seconds")

    # # now we'll do reuters
    # t0 = time()
    # corpus = [" ".join(text) for text in master_object["rt_train"]]
    # t2v = Top2Vec(corpus, speed="learn")
    # topic_words, _, _ = t2v.get_topics()
    # for ref in ["test"]:
    #     key = "rt_" + ref
    #     ref_corpus = master_object[key]
    #     for coherence in ["c_v", "c_npmi"]:
    #         for topn in [5, 10, 20]:
    #             cm = CoherenceModel(topics=topic_words, texts=ref_corpus, dictionary=rt_dict, topn=topn, coherence=coherence)
    #             score = cm.get_coherence()
    #             row = f"rt,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},t2v,na,na,na,na,na,{coherence},{topn},{score}"
    #             f.write(row + "\n")
    # t1 = time()
    # print(f"t2v on rt finished in {t1 - t0} seconds")

    corpus = [" ".join(text) for text in master_object["ep_train"]]
    t0 = time()
    t2v = Top2Vec(corpus, speed="learn")
    t1 = time()
    topic_words, _, _ = t2v.get_topics()
    print(f"Num topics: {len(topic_words)}")
    # print(topic_words)


    for topic in topic_words:
        print(topic[:10])

    print("puw:", proportion_unique_words(topic_words, topk=10))
    print("jd:", pairwise_jaccard_diversity(topic_words, topk=10))
    print("irbo p=0.5:", irbo(topic_words, weight=0.5, topk=10))
    print("irbo p=0.9:", irbo(topic_words, weight=0.9, topk=10))

    for ref in ["test"]:
        key = "ep_" + ref
        ref_corpus = master_object[key]
        for coherence in ["c_v", "c_npmi", "u_mass"]:
            for topn in [5, 10, 20]:
                cm = CoherenceModel(topics=topic_words, texts=ref_corpus, dictionary=ep_dict, topn=topn,
                                    coherence=coherence)
                score = cm.get_coherence()
                print(coherence, ":", score, "(topn=", topn, ")")
                # row = f"bbc,{ref},{ner},{pos_filter},{phrase},{phrase_threshold},t2v,na,na,na,na,na,{coherence},{topn},{score}"
                # f.write(row + "\n")
    print(f"t2v finished in {t1 - t0} seconds")

    # f.close()


if __name__ == "__main__":
    main()
