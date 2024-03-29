import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition

df = pd.read_csv('example_data.csv')
sentences = df['corpus'].tolist()
word_lst = []
for sentence in sentences:
    word_lst.append(sentence.split())
MW2V = gensim.models.word2vec.Word2Vec(word_lst, sg=0)
# MW2V = gensim.models.word2vec.Word2Vec(word_lst, window=100, sg=0)
MW2V.save("XinMinWORD2Vec")

words_set = set(MW2V.wv.index_to_key)
countries = ["巴拿马", "所罗门群岛", "斯洛伐克", "贝宁", "圣多美和普林西比",
             "埃及",
             "中非", "冈比亚", "以色列", "科特迪瓦", "佛得角", "亚美尼亚",
             "波斯尼亚",
             "阿尔巴尼亚", "比利时", "马来西亚", "伊拉克", "苏里南", "津巴布韦",
             "伊朗",
             "布隆迪", "巴勒斯坦", "秘鲁", "立陶宛", "几内亚比绍", "智利",
             "新加坡",
             "卡塔尔", "利比亚", "萨摩亚", "墨西哥", "朝鲜", "缅甸", "柬埔寨",
             "英国",
             "巴西", "阿富汗", "日本", "格鲁吉亚", "巴基斯坦", "爱沙尼亚",
             "孟加拉",
             "毛里塔尼亚", "马尔代夫", "匈牙利", "沙特", "尼日尔", "拉脱维亚",
             "文莱",
             "哈萨克斯坦", "波兰", "安道尔", "卢森堡", "塞拉利昂", "阿曼",
             "台湾",
             "印度", "毛里求斯", "斯洛文尼亚", "韩国", "古巴", "希腊", "蒙古",
             "纳米比亚", "乍得", "摩纳哥", "埃塞俄比亚", "丹麦", "挪威",
             "哥伦比亚",
             "格林纳达", "摩洛哥", "德国", "斯里兰卡", "苏丹", "汤加",
             "澳大利亚",
             "新西兰", "叙利亚", "突尼斯", "刚果金", "阿根廷", "阿尔及利亚",
             "南非",
             "奥地利", "乌干达", "特立尼达和多巴哥", "喀麦隆", "塞舌尔",
             "葡萄牙",
             "保加利亚", "不丹", "东帝汶", "乌拉圭", "委内瑞拉", "瑞士",
             "玻利维亚",
             "西班牙", "摩尔多瓦", "加纳", "土库曼斯坦", "圭亚那", "吉尔吉斯",
             "坦桑尼亚", "尼日利亚", "塔吉克斯坦", "乌兹别克斯坦", "阿联酋",
             "马里",
             "瑞典", "白俄罗斯", "多哥", "法国", "罗马尼亚", "圣卢西亚",
             "俄罗斯",
             "赞比亚", "加蓬", "科威特", "卢旺达", "几内亚", "塞内加尔",
             "赤道几内亚",
             "泰国", "瑙鲁", "厄瓜多尔", "老挝", "荷兰", "马耳他", "越南",
             "尼泊尔",
             "博茨瓦纳", "利比里亚", "约旦", "多米尼克", "爱尔兰", "也门",
             "安哥拉",
             "吉布提", "巴林", "瓦努阿图", "土耳其", "美国", "刚果布",
             "塞浦路斯",
             "冰岛", "莱索托", "巴哈马", "意大利", "菲律宾", "索马里", "印尼",
             "阿塞拜疆", "肯尼亚", "巴巴多斯", "牙买加", "塞尔维亚",
             "列支敦士登",
             "密克罗尼西亚", "马其顿", "新几内亚", "黎巴嫩", "斐济", "莫桑比克",
             "厄立特里亚", "圣马力诺", "布基纳法索", "捷克", "芬兰", "科摩罗",
             "克罗地亚", "加拿大", "安提瓜和巴布达", "马达加斯加", "乌克兰",
             "图瓦卢",
             "圣文森特和格林纳丁斯", "多米尼加", "哥斯达黎加", "基里巴斯",
             "斯威士兰",
             "巴拉圭", "帕劳", "马拉维", "萨尔瓦多", "尼加拉瓜", "海地",
             "南苏丹",
             "伯利兹", "危地马拉", "洪都拉斯", "黑山共和国", "圣基茨和尼维斯",
             "梵蒂冈",
             "马绍尔群岛"]
target_words = list(set(countries) & words_set)
wordsSubMatrix = []
for word in target_words:
    wordsSubMatrix.append(MW2V.wv[word])
wordsSubMatrix = np.array(wordsSubMatrix)
pcaWords = sklearn.decomposition.PCA(n_components=55).fit(wordsSubMatrix)
reducedPCA_data = pcaWords.transform(wordsSubMatrix)
# tsneWords = sklearn.manifold.TSNE(n_components=2).fit_transform(reducedPCA_data)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(reducedPCA_data[:, 0], reducedPCA_data[:, 1], alpha=0)
for i, word in enumerate(target_words):
    ax.annotate(word, (reducedPCA_data[:, 0][i], reducedPCA_data[:, 1][i]),
                size=20 * (55 - i) / 55)
plt.xticks(())
plt.yticks(())
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.savefig('countries pca.png')
plt.show()

