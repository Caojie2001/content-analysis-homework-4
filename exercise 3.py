import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise
import sklearn.manifold
import sklearn.decomposition

df = pd.read_csv('example_data.csv')
sentences = df['corpus'].tolist()
word_lst = []
for sentence in sentences:
    word_lst.append(sentence.split())
# MW2V = gensim.models.word2vec.Word2Vec(word_lst, sg=0)
# MW2V.save("XinMinWORD2Vec")
MW2V = gensim.models.word2vec.Word2Vec.load('XinMinWORD2Vec')
words_set = set(MW2V.wv.index_to_key)


def normalize(vector):
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector


def dimension(model, positives, negatives):
    diff = sum([normalize(model[x]) for x in positives]) - sum(
        [normalize(model[y]) for y in negatives])
    return diff


def gen_word_list(lst, wset):
    return list(set(lst) & wset)


professions = ["老师", "护士", "工程师", "警察",
               "律师", "会计", "建筑师", "厨师", "服务员", "司机", "销售",
               "经理", "研究员", "设计师", "作家"]

target_words = list(set(professions) & words_set)
Gender = dimension(MW2V.wv,
                   gen_word_list(['他', '男人', '男孩', '他们'], words_set),
                   gen_word_list(['她', '她们', '女孩'], words_set))
Class = dimension(MW2V.wv, gen_word_list(
    ["富裕", "财富", "富有", "富足", "财产", "资产"], words_set), gen_word_list(
    ["贫穷", "贫困", "穷困", "贫乏", "困难", "穷人"], words_set))


def makeDF(model, word_list):
    g = []
    c = []
    for word in word_list:
        g.append(sklearn.metrics.pairwise.cosine_similarity(
            MW2V.wv[word].reshape(1, -1), Gender.reshape(1, -1))[0][0])
        c.append(sklearn.metrics.pairwise.cosine_similarity(
            MW2V.wv[word].reshape(1, -1), Class.reshape(1, -1))[0][0])
    df = pd.DataFrame({'gender': g, 'class': c}, index=word_list)
    return df


OCCdf = makeDF(MW2V, professions)


def Coloring(Series):
    x = Series.values
    y = x - x.min()
    z = y / y.max()
    c = list(plt.cm.rainbow(z))
    return c


def PlotDimension(ax, df, dim):
    ax.set_frame_on(False)
    ax.set_title(dim, fontsize=20)
    colors = Coloring(df[dim])
    for i, word in enumerate(df.index):
        ax.annotate(word, (0, df[dim][i]), color=colors[i], alpha=0.6,
                    fontsize=12)
    MaxY = df[dim].max()
    MinY = df[dim].min()
    plt.ylim(MinY, MaxY)
    plt.yticks(())
    plt.xticks(())
for i in professions:
    print(i)
'''
fig = plt.figure(figsize=(12, 4))

# 创建第一个子图
ax1 = fig.add_subplot(121)  # 1行2列的网格中的第1个
PlotDimension(ax1, OCCdf, 'gender')

# 创建第二个子图（如果需要）
ax2 = fig.add_subplot(122)  # 1行2列的网格中的第2个
# 假设有另一个维度比如 'class' 可以这样画：
PlotDimension(ax2, OCCdf, 'class')

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.savefig('profession.png')
plt.show()
'''