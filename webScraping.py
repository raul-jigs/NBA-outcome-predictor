from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

#Player Statustics per matching scraping
def scrape_basic_player_stats(begin_year, end_year):
    for i in range(end_year, begin_year - 1, -1):
        data = []
        driver = webdriver.Chrome(ChromeDriverManager().install())
        #url of the page we want to scrape 
        url= "https://www.nba.com/stats/players/boxscores/?Season=20"+str(i//10)+str(i%10)+"-" + str((i+1)//10)+ str((i+1)%10) + "&SeasonType=Regular%20Season"
        print(url)
        # initiating the webdriver. Parameter includes the path of the webdriver. 
        driver.get(url)  
        time.sleep(10) 
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        stats_block = soup.find("section", {"class": "nba-stats-content-block"})
        pages= soup.find("div", class_="Pagination_content__f2at7 Crom_cromSetting__Tqtiq").find("select", {"class": "DropDown_select__4pIg9"}).find_all("option")

        heading= stats_block.find("table", {"class": "Crom_table__p1iZz"}).find("tr", {"class": "Crom_headers__mzI_m"})
        headers = [header["field"] for header in heading.find_all("th")][:-1]
        for page in range(len(pages)-1):
            html = driver.page_source 
            soup = BeautifulSoup(html, "html.parser")
            table = stats_block.find("table", {"class": "Crom_table__p1iZz"}).find("tbody", {"class": "Crom_body__UYOcU"})
        
            heading= stats_block.find("table", {"class": "Crom_table__p1iZz"}).find("tr", {"class": "Crom_headers__mzI_m"})
            headers = [header["field"] for header in heading.find_all("th")][:-1]
            tr= table.find_all('tr')
            for row in tr:
                td= row.find_all('td')[:-1]
                if len(td) != 0:
                    data_point = [((col.text).replace('\n', '')).strip() for col in td]
                    data_point = ["20"+str(i//10)+str(i%10)+"-"+str((i+1)//10)+str((i+1)%10) if text=='' else text for text in data_point]
                    data.append(data_point)
            nextButton= driver.find_elements("xpath", '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[5]/button[2]')[0]
            driver.execute_script("arguments[0].click();", nextButton)
            time.sleep(5)
        pd.DataFrame(data).to_csv("/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/PlayerSchedule"+str(i//10)+str(i%10)+"-" + str((i+1)//10)+ str((i+1)%10)+".csv", header=headers, index=None)
        driver.close() # closing the webdriver

def scrape_advanced_player_stats(begin_year, end_year):
    for i in range(end_year, begin_year - 1, -1):
        data = []
        driver = webdriver.Chrome(ChromeDriverManager().install())
        #url of the page we want to scrape 
        url = "https://www.nba.com/stats/players/boxscores-advanced?Season=20" + str(str(i//10)+str(i%10))+"-" + str((i+1)//10)+ str((i+1)%10) + "&SeasonType=Regular%20Season"
        driver.get(url)  
        time.sleep(10) 
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        stats_block = soup.find("section", {"class": "nba-stats-content-block"})
        pages= soup.find("div", class_="Pagination_content__f2at7 Crom_cromSetting__Tqtiq").find("select", {"class": "DropDown_select__4pIg9"}).find_all("option")

        heading= stats_block.find("table", {"class": "Crom_table__p1iZz"}).find("tr", {"class": "Crom_headers__mzI_m"})
        headers = [header["field"] for header in heading.find_all("th")][:-1]
        for page in range(len(pages)-1):
            html = driver.page_source 
            soup = BeautifulSoup(html, "html.parser")
            table = stats_block.find("table", {"class": "Crom_table__p1iZz"}).find("tbody", {"class": "Crom_body__UYOcU"})
        
            heading= stats_block.find("table", {"class": "Crom_table__p1iZz"}).find("tr", {"class": "Crom_headers__mzI_m"})
            headers = [header["field"] for header in heading.find_all("th")][:-1]
            tr= table.find_all('tr')
            for row in tr:
                td= row.find_all('td')[:-1]
                if len(td) != 0:
                    data_point = [((col.text).replace('\n', '')).strip() for col in td]
                    data_point = ["20"+str(i//10)+str(i%10)+"-"+str((i+1)//10)+str((i+1)%10) if text=='' else text for text in data_point]
                    data.append(data_point)
            nextButton= driver.find_elements("xpath", '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[5]/button[2]')[0]
            driver.execute_script("arguments[0].click();", nextButton)
            time.sleep(5)
        pd.DataFrame(data).to_csv("/Users/raulalcantar/Downloads/NBA_Game_Predictor-main/data/AdvancedPlayerStats"+str(i//10)+str(i%10)+"-" + str((i+1)//10)+ str((i+1)%10)+".csv", header=headers, index=None)
        driver.close() # closing the webdriver

