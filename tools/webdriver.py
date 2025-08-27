from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time

# 设置Chrome选项以在后台运行
# options = Options()
# options.headless = True
mobile_emulation = {
    "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 2.0},
    "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1"
}

options = Options()
options.add_argument('--headless')  # 无头模式
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_experimental_option('mobileEmulation', mobile_emulation)
# options.add_argument("--window-size=1920,1080")

# 设置Desired Capabilities以模拟iPhone 6
# desired_capabilities = DesiredCapabilities.CHROME
# desired_capabilities['goog:chromeOptions'] = {'mobileEmulation': {'deviceName': 'iPhone 13'}}

# 配置driver本地路径
driverpath = "C:\Program Files\Google\Chrome\Application\chromedriver.exe"
service = Service(executable_path=driverpath)

# 创建WebDriver实例
driver = webdriver.Chrome(service=service, options=options)

# 打开链接
driver.get("https://translate.volcengine.com/")

# 等待页面加载完成
time.sleep(5)

# 截图并保存
driver.save_screenshot("screenshot.png")

# 关闭浏览器
driver.quit()
