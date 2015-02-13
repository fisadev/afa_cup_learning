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

YEAR_URL = 'http://www.rsssf.com/tablesa/arg%s.html'
FILE_PATH = 'data/%i.txt'
YEARS = {}

# 1990-1999: files with format "arg90.html"
for year in range(1990, 2000):
    YEARS[year] = str(year - 1900)

# 2000-2009: files with format "arg00.html"
for year in range(2000, 2010):
    YEARS[year] = str(year - 2000).rjust(2, '0')

# 2010-2015: files with format "arg2010.html"
for year in range(2010, 2016):
    YEARS[year] = str(year)


for year, year_in_file in YEARS.items():
    url = YEAR_URL % year_in_file
    print 'Year:', year, '(', url, ')'
    data = requests.get(url).content
    data = data.decode('windows-1252')
    data = data.replace('\r\n', '\n')
    data = html2text(data)
    data = data.encode('utf-8')

    with open(FILE_PATH % year, 'w') as data_file:
        data_file.write(data)

    print 'Wrote file with', len(data), 'chars'
