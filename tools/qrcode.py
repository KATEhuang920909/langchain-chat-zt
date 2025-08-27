import cv2, os
import pyzbar.pyzbar as pyzbar
import imageio.v2 as imageio
import validators
from pydantic import BaseModel, Field
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from bs4 import BeautifulSoup
import re, os, requests

tmp_qr_path = "/home/ding/LLM/upload/"

def read_qr_code():
    # 获取目录下的所有文件和目录名
    entries = os.listdir(tmp_qr_path)

    # 过滤出所有的文件
    file_names = [entry for entry in entries if os.path.isfile(os.path.join(tmp_qr_path, entry))]

    # 粗暴！！！！！只取第一个
    file_name = file_names[0]

    img = imageio.imread(tmp_qr_path + file_name)

    #转化为灰度
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 扫描二维码
    texts = pyzbar.decode(gray_img)
    print(texts)
    if texts==[]:
        print("未识别成功")
        return 'www.baidu.com'
    else:
        for text in texts:
            res = text.data.decode("utf-8")
        print("识别成功")
        print(res)
    
        return res

    # # 测试二维码信息是否为网址
    # def is_url(string):
    #     return validators.url(string)


    # test_string = res
    # print(is_url(test_string))  # 应该返回 True

# 定义网页分析
def get_searchlist(url):
    options = Options()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--lang=zh-CN") # 将浏览器标识为中文，否则会出现翻译问题
    # 配置driver本地路径
    driverpath = "/usr/bin/chromedriver"
    service = Service(executable_path=driverpath)
    # 创建WebDriver实例
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 3)
    # 获取页面源代码
    html = driver.page_source
    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html, 'html.parser')

    webcontent = soup.get_text()
    driver.quit()
    return webcontent