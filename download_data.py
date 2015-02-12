# coding: utf-8
"""
Script to download the raw data from http://www.rsssf.com/
The data was processed mostly by interactive sessions in ipython. Almost every
file had it's own format, so there is no point in trying to automate it in a
fully automatic script, but this downloading script may be useful for future
dowloads.
"""
from html2text import html2text
import requests

YEAR_URL = 'http://www.rsssf.com/tablesa/arg%i.html'
FILE_PATH = 'data/%i.txt'
YEARS = range(90, 100) + range(2000, 2016)

for year in YEARS:
    print 'Year:', year
    data = requests.get(YEAR_URL % year).content
    data = data.decode('windows-1252')
    data = data.replace('\r\n', '\n')
    data = html2text(data)
    data = data.encode('utf-8')

    with open(FILE_PATH % year, 'w') as data_file:
        data_file.write(data)

    print 'Wrote file with', len(data), 'chars'
