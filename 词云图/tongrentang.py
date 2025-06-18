from matplotlib import colors
import imageio
import wordcloud
import jieba
import pandas as pd

csv_file_path = './同仁堂评论.csv'
df = pd.read_csv(csv_file_path)

data = ' '.join(df['评价内容'].astype(str))

wordslist = [word for word in jieba.lcut(data) if len(word) >= 2]

stopwords = []
with open('./呼呼停用.txt', encoding='utf8') as fs:
    for line in fs:
        stopwords.append(line.strip('\n'))
word_count = {}
for word in wordslist:
    if word in stopwords:
        continue
    word_count[word] = word_count.get(word, 0)+1


maskdata = imageio.v2.imread('./遮罩.png')

color_list = ['#D5B48A', '#A6944A', '#858FAC', '#CFA168', '#C9842E']
color_map = colors.ListedColormap(color_list)

c = wordcloud.WordCloud(font_path='./YanZhenQingDuoBaoTaBei-2.ttf', width=800, height=600,
                        background_color='white', mask=maskdata,
                        colormap=color_map)

c.generate_from_frequencies(word_count)

c.to_file('同仁堂.png')
