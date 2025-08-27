from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


options = Options()
options.add_argument('--headless')  # 无头模式
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# 配置driver本地路径
driverpath = "/usr/bin/chromedriver"
service = Service(executable_path=driverpath)

# 创建WebDriver实例
driver = webdriver.Chrome(service=service, options=options)

url = 'https://appgallery.huawei.com/search/%E6%8A%96%E9%9F%B3'
driver.get(url)

wait = WebDriverWait(driver, 3)
element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.first_lowertexts')))

# 获取页面源代码
html = driver.page_source

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html, 'html.parser')

# print(soup)
labels = soup.find_all('p', attrs={'data-v-302a9de2': True})
# print(labels)
results = []
for label in labels:
    results.append(label.text)

results = results[::2]
print(results)