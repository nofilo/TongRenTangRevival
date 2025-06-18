import requests
import random
import time
from bs4 import BeautifulSoup as bs
import pymysql
from urllib.parse import urlparse
head = {'user-agent':'Mozilla/5.0'}#模拟浏览器访问

# 确定爬取的数据
def getData():
    try:
        #获取网址
        for i in range(0,14):
            i = i+1
            new_url = "https://book.douban.com/latest?subcat=全部&p=" + str(i)
            parsed_url = urlparse(new_url)
            r =  requests.get(parsed_url.geturl(),headers = head)
            time.sleep(random.randrange(1,3))
            r.encoding = r.apparent_encoding
            soup1 = bs(r.text,'html.parser')
            soup = soup1.find_all('li',attrs={'class',"media clearfix"}) #and soup1.find_all('li',attrs={'class',"media clearfix last"}) 
            for w in soup:
                #书籍名字
                name = w.find_next('h2',attrs={'class',"clearfix"}).get_text().strip()               
                #作者
                A = str(w.find_next('p',attrs={'class',"subject-abstract color-gray"}).get_text().strip())
                aa = A.find('/ 20')
                author = A[:aa]
                #图片url
                picture = w.find_next('div',attrs={'class',"media__img"}).img.attrs['src']
                #出版时间
                ab = A.find('/ 20') 
                Time = A[ab+2:ab+8]
                #装订精度
                version = A[-2:]
                #评分
                S = str(w.find('span',attrs={'font-small color-red fleft'}))
                s = S.replace('<span class="font-small color-red fleft"></span>','<span class="font-small color-red fleft">没有评分​</span>')
                s1 = s.find('fleft">')
                e1 = s.find('</span>')
                score = s[s1+7:e1]               
                #评价人数
                C = w.find('span',attrs = {'fleft ml8 color-gray'}).get_text().strip()
                count = C[1:-4]
                
                saveData(name, author,picture,Time,version,score,count)
    except Exception as E:
            print('Error',E)
            
def saveData(name, author,picture,Time,version,score,count):
    try:
        db = pymysql.connect(host = 'localhost',
                            user = 'root',
                            password='12345678',
                            database='cucer1',
                            charset='utf8')
        cursor = db.cursor()
        sql = 'insert into db(name, author,picture,time,version,score,count_people) values (%s,%s,%s,%s,%s,%s,%s)'
        cursor.execute(sql,(name, author,picture,Time,version,score,count))
        db.commit()
        print('save success')
    except Exception as e:
        print('db-Error',e)
        db.rollback()
    db.close()


getData() 
