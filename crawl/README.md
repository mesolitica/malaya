# How-to-Crawl

**_Last update 3-July-2019, still usable._**

1. Install dependencies

Malaya must installed first.

For ubuntu / debian based
```bash
pip3 install bs4 newspaper3k fake_useragent unidecode
apt-get install libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev libpng12-dev -y
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3
```

For Mac OS
```bash
brew install libxml2 libxslt
brew install libtiff libjpeg webp little-cms2
pip3 install bs4 newspaper3k fake_useragent unidecode
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3
```

2. Run main.py

```bash
python3 main.py -i "isu mahathir" -s 2009 -e 2019 -l 10
```

You can read more about crawler in [Malaya Wiki](https://github.com/DevconX/Malaya/wiki).

## Issues crawled

You can get download some crawled data from [here](https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/crawler-data.zip)

Last updated (15th August 2018), crawled until the end of google results.

1. isu 1mdb
2. isu agama
3. isu agong
4. isu agrikulture
5. isu air
6. isu anwar ibrahim
7. isu artis
8. isu astro
9. isu bahasa melayu
10. isu barisan nasional
11. isu cikgu
12. isu cukai
13. isu cyberjaya
14. isu dunia
15. isu ekonomi
16. isu gst
17. isu harakah
18. isu harga
19. isu icerd
20. isu imigren
21. isu kapitalis
22. isu kerajaan
23. isu kesihatan
24. isu kuala lumpur
25. isu lgbt
26. isu mahathir
27. isu makanan
28. isu malaysia airlines
29. isu malaysia
30. isu minyak
31. isu najib razak
32. isu pelajar
33. isu pelakon
34. isu pembangkang
35. isu perkauman
36. isu permainan
37. isu pertanian
38. isu politik
39. isu rosmah
40. isu sabah
41. isu sarawak
42. isu sekolah
43. isu sosial media
44. isu sultan melayu
45. isu teknologi
46. isu tm
47. isu ubat
48. isu universiti
49. isu wan azizah
50. peluang pekerjaan
51. perkahwinan
