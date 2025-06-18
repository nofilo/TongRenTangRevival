from matplotlib import colors
import imageio
import wordcloud
import jieba
import pandas as pd

xls_file_path = './28题文件.xlsx'
df = pd.read_excel(xls_file_path)
data = ' '.join(df['q28'].astype(str))


wordslist = [word for word in jieba.lcut(data) if len(word) >= 2]

stopwords = []
with open('.\\呼呼停用.txt', encoding='utf8') as fs:
    for line in fs:
        stopwords.append(line.strip('\n'))
word_count = {}
for word in wordslist:
    if word in stopwords:
        continue
    word_count[word] = word_count.get(word, 0)+1

maskdata = imageio.v2.imread('./遮罩.jpg')

color_list = ['#D5B48A', '#A6944A', '#858FAC', '#CFA168', '#C9842E']
color_map = colors.ListedColormap(color_list)

c = wordcloud.WordCloud(font_path='./YanZhenQingDuoBaoTaBei-2.ttf', width=800, height=600,
                        background_color='white', mask=maskdata,
                        colormap=color_map)

c.fit_words(word_count)

c.to_file('消费者文本.png')
