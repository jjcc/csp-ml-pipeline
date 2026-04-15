from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import platform

url_vix = "https://www.cboe.com/tradable_products/vix/"

XPATH_VIX_NEW = '/html/body/main/div/div/section[1]/div/div[1]/div[2]/div[1]/h2'
# Set up the Chrome driver automatically
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

def get_current_vix(url, driver):
    vix_value = None
    try:
        driver.get(url)

    # Wait up to 10 seconds for the element to be present
        wait = WebDriverWait(driver, 10)
        #vix_element = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="charts-tile"]/div/div/div[1]/div[1]/div[2]')))
        vix_element = wait.until(EC.presence_of_element_located((By.XPATH, XPATH_VIX_NEW)))

        vix_value = vix_element.text.strip()
        vix_value = vix_value.replace("$", "")
        print("Current VIX value:", vix_value)

    except Exception as e:
        print("Error fetching VIX value:", str(e))

    finally:
        driver.quit()
    return vix_value


def init_driver(headless=True):
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service as ChromeService
    chrome_options = webdriver.ChromeOptions()
    if headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--ignore-certificate-errors-spki-list')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--window-size=1200,900')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('disable-infobars')
                 #Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    chrome_options.add_argument('user-agent={0}'.format(user_agent))

    options = {
        'exclude_hosts': ['*.cloudfront.net', 'www.google-analytics.com']  # Bypass Selenium Wire for these hosts
    }

    platform_name = platform.system()
    if platform_name == 'Windows':
        #mydriver = webdriver.Chrome( r'D:/Software/WebDriver/chromedriver.exe', seleniumwire_options=options) #options=options)
        mydriver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),
                                    options = chrome_options)
    else:
        user_data_dir = r'/home/jchen/.config/google-chrome/default'
        chrome_options.add_argument(f"user-data-dir={user_data_dir}")
        mydriver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),
                                    options = chrome_options)
    return mydriver
#vix = get_current_vix(url, driver)
