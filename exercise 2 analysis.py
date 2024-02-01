import gensim
import pandas as pd
from gensim.models.doc2vec import TaggedDocument

df = pd.read_csv('example_data.csv', index_col=0)

countries = ['巴基斯坦', '委内瑞拉', '菲律宾', '黎巴嫩', '韩国', '瑞典', '波兰',
             '加拿大', '奥地利', '斯里兰卡', '泰国', '新加坡', '越南', '芬兰',
             '新西兰', '澳大利亚', '德国', '俄罗斯', '阿根廷', '法国',
             '塞尔维亚', '尼日利亚', '丹麦', '印度', '土耳其', '台湾', '朝鲜',
             '哥伦比亚', '伊朗', '巴西', '日本', '瑞士', '墨西哥', '乌克兰',
             '沙特', '阿联酋', '希腊', '叙利亚', '古巴', '荷兰', '伊拉克',
             '比利时', '葡萄牙', '埃塞俄比亚', '美国', '意大利', '西班牙',
             '以色列', '哈萨克斯坦', '英国', '巴勒斯坦', '卡塔尔', '巴拿马',
             '挪威', '阿富汗']

D2V = gensim.models.word2vec.Word2Vec.load('D2V')
'''
word_vector = D2V['美国']
for i in D2V.docvecs.most_similar([word_vector], topn=10):
    print(i)
'''
'''
countries = ['墨西哥', '法国', '意大利', '德国', '西班牙', '英国', '巴西']
heatmapMatrix = []
for tagOuter in countries:
    column = []
    tagVec = D2V.docvecs[tagOuter].reshape(1, -1)
    for tagInner in countries:
        column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, 
        D2V.docvecs[tagInner].reshape(1, -1))[0][0])
    heatmapMatrix.append(column)
heatmapMatrix = np.array(heatmapMatrix)
fig, ax = plt.subplots()
hmap = ax.pcolor(heatmapMatrix, cmap='terrain')
cbar = plt.colorbar(hmap)

cbar.set_label('cosine similarity', rotation=270)
a = ax.set_xticks(np.arange(heatmapMatrix.shape[1]) + 0.5, minor=False)
a = ax.set_yticks(np.arange(heatmapMatrix.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(countries, minor=False, rotation=270, fontsize=6)
ax.set_yticklabels(countries, minor=False, fontsize=6)
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.savefig('heatmap_euro.png')
plt.show()
'''
print(D2V.docvecs.most_similar([D2V['西班牙']], topn=30))