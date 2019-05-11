# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:51:11 2016

@author: Joanna
"""

import urllib
import re
from bs4 import BeautifulSoup
import mechanicalsoup
import tempfile
import os
import requests


urls = ["http://www.google.com", "http://www.yahoo.com.br"]

regex = str('<title>(.+?)</title>')
pattern = re.compile (regex)

form_html = """
<form method="post" action=historical_data&curr_id=97&st_date=05%2F02%2F2016&end_date=05%2F02%2F2016&interval_sec=Daily>
"""

#for url in urls:
#    htmlfile = urllib.request.urlopen (url)
#    htmltext = str(htmlfile.read ())
#    titles = re.findall (pattern, htmltext)
    
#    print (titles)

br = mechanicalsoup.Browser ()


url = 'http://www.investing.com/instruments/HistoricalDataAjax'

#response = br.submit(form, result.url)

#result = br.get(url)

#headers = POST /instruments/HistoricalDataAjax HTTP/1.1

headers = {
            'Host': 'www.investing.com',
            'Connection': 'keep-alive',
            'Content-Length': '98',
            'Accept': 'text/plain, */*; q=0.01',
            'Origin': 'http://www.investing.com',
            'X-Requested-With': 'XMLHttpRequest',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'http://www.investing.com/currencies/eur-usd-historical-data',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.8',
            'Cookie': 'PHPSESSID=3h19o6k2ittsdkks0brlq4dj81; SideBlockUser=a%3A2%3A%7Bs%3A10%3A%22stack_size%22%3Ba%3A1%3A%7Bs%3A11%3A%22last_quotes%22%3Bi%3A8%3B%7Ds%3A6%3A%22stacks%22%3Ba%3A1%3A%7Bs%3A11%3A%22last_quotes%22%3Ba%3A1%3A%7Bi%3A0%3Ba%3A3%3A%7Bs%3A7%3A%22pair_ID%22%3Bs%3A1%3A%221%22%3Bs%3A10%3A%22pair_title%22%3Bs%3A14%3A%22Euro+US+Dollar%22%3Bs%3A9%3A%22pair_link%22%3Bs%3A19%3A%22%2Fcurrencies%2Feur-usd%22%3B%7D%7D%7D%7D; geoC=PL; fpros_popup=up; gtmFired=OK; optimizelyEndUserId=oeu1467564264778r0.7897276133136657; optimizelySegments=%7B%224225444387%22%3A%22gc%22%2C%224226973206%22%3A%22search%22%2C%224232593061%22%3A%22false%22%2C%225010352657%22%3A%22none%22%7D; optimizelyBuckets=%7B%7D; show_big_billboard1=true; _gat=1; _gat_allSitesTracker=1; _ga=GA1.2.1052885457.1467564277; __gads=ID=683d56bfd1410cce:T=1467564285:S=ALNI_MYSrqVbw4eYcmivmRJ3hhfOzoMmOQ; notification_analysis_200139640=9935; __qca=P0-701037908-1467564295544'
        }

for currency_no in range (100,200):
    print (currency_no)
    for year in range (1970,2017):
        print(year)        
        payload = {'action': 'historical_data',
                   'curr_id':str(currency_no),
                   'st_date':'01/01/'+str(year),
                   'end_date':'12/31/'+str(year),
                   'interval_sec':'Daily'}
        
        r = requests.post(url, data=payload, headers=headers)
        
        with open("ccy_hist_"+str(currency_no)+".txt", "a") as myfile:
            myfile.write("ccy_id: "+str(currency_no))
            myfile.write("year: "+str(year))
            myfile.write(str(r.text[10:]))
        print(r.text)


