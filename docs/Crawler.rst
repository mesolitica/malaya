Crawler
============

There is no official compiled package for the Crawler inside Malaya, but it's included in the repository.

From Source
-----------

The crawler is actively developed on
`Github <https://github.com/huseinzol05/Malaya/tree/master/crawl>`__.

You need to clone the public repo:

.. code:: bash

    git clone https://github.com/huseinzol05/malaya

You need to install dependencies before able to use the crawler.

For ubuntu / debian based:

.. code:: bash

    pip3 install bs4 newspaper3k fake_useragent unidecode
    apt-get install libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev libpng12-dev -y
    curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3

For Mac OS:

.. code:: bash

    brew install libxml2 libxslt
    brew install libtiff libjpeg webp little-cms2
    pip3 install bs4 newspaper3k fake_useragent unidecode
    curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3

And start the crawler:

.. code:: bash

    python3 crawl/main.py -i "isu mahathir" -s 2009 -e 2019 -l 10

Get Help
--------

You can check help from the crawler:

.. code:: bash

    python3 crawl/main.py --help

.. parsed-literal::

    usage: main.py [-h] -i ISSUE -s START -e END -l LIMIT [-p SLEEP] [-m MALAYA]

    optional arguments:
      -h, --help                  show this help message and exit
      -i ISSUE, --issue ISSUE     issue to search
      -s START, --start START     year start to crawl
      -e END, --end END           year end to crawl
      -l LIMIT, --limit LIMIT     limit of articles to crawl
      -p SLEEP, --sleep SLEEP     seconds to sleep for every 10 articles
      -m MALAYA, --malaya MALAYA  boolean to use Malaya

How to start
------------

.. code:: bash

    python3 crawl/main.py -i "isu mahathir" -s 2009 -e 2019 -l 10

The data will be saved later inside ``crawl/`` in ``.json``

Example result return:

.. code:: json

  {
        "title":"Mahathir alergik isu agama",
        "url":"http://www.utusan.com.my/berita/politik/mahathir-alergik-isu-agama-1.583393",
        "authors":[
           "Muhammad Hasif Idris"
        ],
        "top-image":"http://www.utusan.com.my/polopoly_fs/1.557597!/image/image.jpg_gen/derivatives/landscape_650/image.jpg",
        "text":"KOTA BHARU 2 Jan. \u2013 Pas menyifatkan tindakan Pengerusi Parti Pribumi Bersatu Malaysia (PPBM), Tun Dr. Mahathir Mohamad seolah-olah alergik dengan isu agama kerana sering mengeluarkan kenyataan yang menyerlahkan kejahilannya sendiri.\n\nNaib Presiden Pas, Datuk Mohd. Amar Nik Abdullah berkata, pandangan yang diberikan oleh Dr. Mahathir menunjukkan beliau tidak boleh menerima hakikat sebenar yang berlaku.\n\n\u201cSejak dahulu lagi, bila beliau (Dr. Mahathir) cakap bab agama, tak layak pun, bukan saya hendak merendah-rendahkannya. Namun, beliau tiada kelayakan untuk bercakap, lagi baik diam, apabila bercakap nampak kejahilan diri sendiri.\n\n\u201cMalah, beliau seolah-olah alergik dengan isu agama, apabila memberikan respon nampak keras, macam tidak boleh terima. Saya tidak tahu apa perasaan sebenar beliau sebab sejak dari dahulu lagi dia tidak suka Pas, orang UMNO mana hendak suka Pas,\u201d katanya.\n\nBeliau berkata demikian ketika ditemui pemberita selepas Majlis Amanat Khas Tahun Baharu 2018 dan Perhimpunan Penjawat Awam Kelantan di Kompleks Kota Darul Naim di sini hari ini.\n\nYang turut hadir Menteri Besar, Datuk Ahmad Yakob. - UTUSAN ONLINE",
        "keyword":[
           "kota",
           "nampak",
           "pas",
           "mahathir",
           "sebenar",
           "alergik isu agama"
        ],
        "summary":"beliau ditemui pemberita majlis amanat khas tahun baharu perhimpunan penjawat awam kelantan kompleks kota darul naim. yang hadir menteri besar datuk ahmad yakob. utusan online",
        "news":"Utusan Malaysia",
        "date":"01-01-2018",
        "language":"MALAY"
  }

Parameters
-----------

**issue** : *(string)*

An issue or search you want to crawl, if your search is a sentence, you need to include double quote, ``"isu terkini"``.

**start**: *(int)*

Year start of news to start, eg, ``2009``.

**end**: *(int)*

Year end of news to end, eg, ``2020``.

**limit**: *(int)*

Limit of news want to crawl, eg, if put ``100`` only get more or less than ``100``.

**sleep**: *(int)*

Seconds to let the crawler sleeps to prevent IP block, eg ``10`` represents 10 seconds.

**malaya**: *(bool)*

Boolean to use Malaya, if ``False``, ``summary`` and ``language`` will not returned, but not required Malaya to be install in local machine.
