[https://s3-ap-southeast-1.amazonaws.com/huseinhouse-data/language-detection-data-v4.json](https://s3-ap-southeast-1.amazonaws.com/huseinhouse-data/language-detection-data-v4.json)

You can download any language from [https://dumps.wikimedia.org/mswiki/latest/XXwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/mswiki/latest/XXwiki-latest-pages-articles.xml.bz2), you just need to replace `XX` with language code.

And execute,
```bash
python make-corpus.py XXwiki-latest-pages-articles.xml.bz2 wiki_XX.txt
```
