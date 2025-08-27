# 网页内容爬虫Agent
import os, re 
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

def websiteanalyse(websiteurl: str):
    print(websiteurl)
    webcontent = getwebcontent(websiteurl)
    return webcontent

class WebsiteInput(BaseModel):
    websiteurl: str = Field(description="URL网址")
    # print("地址：" + weburl)


# 定义网页分析
def getwebcontent(url: str):
    options = Options()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--lang=zh-CN") # 将浏览器标识为中文，否则会出现翻译问题
    # 配置driver本地路径
    # driverpath = "/usr/local/bin/chromedriver"
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

    pattern = r'\n{2,}'
    webcontent = re.sub(pattern, '', webcontent)

    return webcontent