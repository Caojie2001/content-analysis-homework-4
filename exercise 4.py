import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn.metrics.pairwise
import sklearn.manifold
import sklearn.decomposition
import scipy.linalg
import copy

df = pd.read_csv('example_data.csv')


def calc_syn0norm(model):
    """since syn0norm is now depricated"""
    return (model.wv.syn0 / np.sqrt((model.wv.syn0 ** 2).sum(-1))[
        ..., np.newaxis]).astype(np.float32)


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison
    between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by
    William Hamilton <wleif@stanford.edu>.
    (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim`
    documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the
    aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the
    vocabulary in words (see `intersection_align_gensim` documentation).
    """
    base_embed = copy.copy(base_embed)
    other_embed = copy.copy(other_embed)
    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed,
                                                              other_embed,
                                                              words=words)

    # get the embedding matrices
    # base_vecs = calc_syn0norm(in_base_embed)
    # other_vecs = calc_syn0norm(in_other_embed)
    base_vecs = [in_base_embed.wv.get_vector(w, norm=True) for w in
                 set(in_base_embed.wv.index_to_key)]
    other_vecs = [in_other_embed.wv.get_vector(w, norm=True) for w in
                  set(in_other_embed.wv.index_to_key)]

    # just a matrix dot product with numpy
    # m = np.array(other_vecs).T.dot(np.array(base_vecs))
    m = np.array(other_vecs, dtype=np.float64).T.dot(
        np.array(base_vecs, dtype=np.float64))
    # SVD method from numpy
    # u, _, v = np.linalg.svd(m)
    u, _, v = scipy.linalg.svd(m, lapack_driver='gesvd')
    # another matrix operation
    ortho = u.dot(v)
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    # other_embed.wv.vectors = (np.array(other_vecs)).dot(ortho)
    other_embed.wv.vectors = (np.array(other_vecs, dtype=np.float64)).dot(ortho)
    other_embed.wv.vectors = other_embed.wv.vectors.astype(np.float32)

    return other_embed


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected
    with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum
    of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both
    gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of
        m2.syn0
        -- you can find the index of any word on the .index2word list:
        model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the
    count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w,
                                                                        "count"),
        reverse=True)

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        new_arr = [m.wv.get_vector(w, norm=True) for w in common_vocab]

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        # old_vocab = m.wv.index_to_key
        new_vocab = []
        k2i = {}
        for new_index, word in enumerate(common_vocab):
            new_vocab.append(word)
            k2i[word] = new_index
        m.wv.index_to_key = new_vocab
        m.wv.key_to_index = k2i
        m.wv.vectors = np.array(new_arr)

    return (m1, m2)


def compareModels(df, category, text_column_name='tokenized_article', sort=True,
                  embeddings_raw={}):
    """If you are using time as your category sorting is important"""
    if len(embeddings_raw) == 0:
        embeddings_raw = rawModels(df, category, text_column_name, sort)
    cats = sorted(set(df[category]))
    # These are much quicker
    embeddings_aligned = {}
    for catOuter in cats:
        embeddings_aligned[catOuter] = [embeddings_raw[catOuter]]
        for catInner in cats:
            embeddings_aligned[catOuter].append(
                smart_procrustes_align_gensim(embeddings_aligned[catOuter][-1],
                                              embeddings_raw[catInner]))
    return embeddings_raw, embeddings_aligned


def rawModels(df, category, text_column_name='tokenized_article', sort=True):
    embeddings_raw = {}
    cats = sorted(set(df[category]))
    for cat in cats:
        # This can take a while
        print("Embedding {}".format(cat), end='\r')
        subsetDF = df[df[category] == cat]
        # You might want to change the W2V parameters
        embeddings_raw[cat] = gensim.models.word2vec.Word2Vec(
            subsetDF[text_column_name].sum())
    return embeddings_raw


rawEmbeddings, comparedEmbeddings = compareModels(df, 'date')


def getDivergenceDF(word, embeddingsDict):
    dists = []
    cats = sorted(set(embeddingsDict.keys()))
    dists = {}
    print(word)
    for cat in cats:
        dists[cat] = []
        for embed in embeddingsDict[cat][1:]:
            dists[cat].append(np.abs(1 -
                                     sklearn.metrics.pairwise.cosine_similarity(
                                         np.expand_dims(
                                             embeddingsDict[cat][0].wv[word],
                                             axis=0),
                                         np.expand_dims(embed.wv[word],
                                                        axis=0))[0, 0]))
    return pd.DataFrame(dists, index=cats)


targetWord = '美国'

pltDF = getDivergenceDF(targetWord, comparedEmbeddings)
fig, ax = plt.subplots(figsize=(10, 7))
seaborn.heatmap(pltDF, ax=ax,
                annot=False)  # set annot True for a lot more information
ax.set_xlabel("Starting day")
ax.set_ylabel("Final day")
ax.set_ylabel("Final day")
ax.set_title("daily linguistic change for: '{}'".format(targetWord))
plt.show()
