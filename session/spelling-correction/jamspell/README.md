# JamSpell

## how-to

1. Make JamSpell,

```bash
git clone https://github.com/bakwc/JamSpell.git
cd JamSpell
mkdir build
cd build
cmake ..
make
```

2. Combined multiple txts, took from https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean

```bash
cat filtered-dumping-wiki.txt dumping-news.txt > combined.txt
```

3. Create JamSpell model,

```bash
./main/jamspell train /home/husein/pure-text/alphabet.txt /home/husein/pure-text/combined.txt out.bin
./main/jamspell train /home/husein/pure-text/alphabet.txt /home/husein/pure-text/filtered-dumping-wiki.txt wiki.bin
./main/jamspell train /home/husein/pure-text/alphabet.txt /home/husein/pure-text/dumping-news.txt news.bin
```