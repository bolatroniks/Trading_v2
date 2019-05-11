# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 21:07:39 2016

@author: Joanna
"""

#import urllib
import re
from bs4 import BeautifulSoup
import mechanicalsoup
import tempfile
import os
#import requests

regex_date = str('<td class="first left bold noWrap">(.+?)</td>')
regex_close = str('<td class="colourFont">(.+?)</td>')
regex_px = str('<td>(.+?)</td>')
                #<td>(.+?)</td>\n
                #<td>(.+?)</td>\n
regex_chg = str('<td class="bold colourFont">(.+?)</td>')
pattern_date = re.compile (regex_date)
pattern_close = re.compile (regex_close)
pattern_chg = re.compile (regex_chg)
pattern_px = re.compile (regex_px)

p = re.compile('(green|red)')

all_cells = []

for currency_no in range (1,117):
    print (currency_no)    
    
    with open("ccy_hist_"+str(currency_no)+".txt", "r") as myfile:
        raw_text = myfile.read()
        soup = BeautifulSoup(raw_text[34:])
        tables = soup.findAll("table", { "class" : "genTbl closedTbl historicalTbl" })
        with open("ccy_hist_ext_"+str(currency_no)+".txt", "w") as my_write_file:

            for table in tables:
                for row in table.findAll("tr"):
                    cells = row.findAll("td")
                    cells = p.subn('colour', str(cells))
                    #for cell in cells:
                    c_date = pattern_date.findall(str(cells))
                    c_close = pattern_close.findall(str(cells))
                    c_px = pattern_px.findall(str(cells))
                    c_chg = pattern_chg.findall(str(cells))
                    #print(str(cells))
                    if len(str(c_date))>5:
                        #print(str(cells))
                        try:
                            my_write_file.write(str(datetime.strptime(c_date[0],"%b %d, %Y").strftime("%d/%m/%Y"))+','+
                                                str(c_close[0])+','+
                                                str(c_px[0])+','+
                                                str(c_px[1])+','+
                                                str(c_px[2])+','+
                                                str(c_chg[0])+'\n')
                        except:
                            print ('exception: '+str(cells))
                    #all_cells.append(cells)
                    #print (cells)
                    