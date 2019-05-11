# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:59:12 2016

@author: Joanna
"""

import sys  
from PyQt5.QtGui import *  
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *  
from PyQt5.QtWebKitWidgets import *  
from lxml import html 

#Take this class for granted.Just use result of rendering.
class Render(QWebPage):  
  def __init__(self, url):  
    self.app = QApplication(sys.argv)  
    QWebPage.__init__(self)  
    self.loadFinished.connect(self._loadFinished)  
    self.mainFrame().load(QUrl(url))  
    self.app.exec_()  
  
  def _loadFinished(self, result):  
    self.frame = self.mainFrame()  
    self.app.quit()  

url = 'http://www.investing.com/currencies/gbp-try-historical-data'  
r = Render(url)  
result = r.frame.toHtml()
#This step is important.Converting QString to Ascii for lxml to process
archive_links = html.fromstring(str(result))
print (archive_links)