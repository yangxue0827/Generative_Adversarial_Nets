# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import requests
from bs4 import BeautifulSoup
import os
import traceback


def download(url, filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                # filter out keep-alive new chunks
                if chunk:
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    if os.path.exists('imgs') is False:
        os.makedirs('imgs')

    start = 1
    end = 8000
    for i in range(start, end + 1):
        url = 'http://konachan.net/post?page=%d&tags=' % i
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        for img in soup.find_all('img', class_="preview"):
            target_url = 'http:' + img['src']
            filename = os.path.join('imgs', target_url.split('/')[-1])
            download(target_url, filename)
        print('%d / %d' % (i, end))


















