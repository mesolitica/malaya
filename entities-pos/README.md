# Bahasa-Entities-POS-Recognition
Use deep learning models to classify entities and POS

## Entities supported
```text
PRN - person, group of people, believes, etc
LOC - location
NORP - Military, police, government, Parties, etc
ORG - Organization, company
LAW - related law document, etc
ART - art of work, special names, etc
EVENT - event happen, etc
FAC - facility, hospitals, clinics, etc
TIME - date, day, time, etc
O - not related, out scope
```

## POS supported
```
KT - Kata Tanya
KJ - Kata Kerja
KP - Kata Perintah
KPA - Kata Pangkal
KB - Kata Bantu
KPENGUAT - Kata Penguat
KPENEGAS - Kata Penegas
NAFI - Kata Nafi
KPEMERI - Kata Pemeri
KS - Kata Sendi
KPEMBENAR - Kata Pembenar
NAFI - Kata Nafi
NO - Numbers
SUKU - Suku Bilangan
PISAHAN - Kata Pisahan
KETERANGAN - Kata Keterangan
ARAH - Kata Arah
KH - Kata Hubung
GN - Ganti Nama
KA - Kata Adjektif
O - not related, out scope
```

## Models used
1. LSTM + CRF + chars embeddings + Static Bidirectional
2. LSTM + chars sequence + Static Bidirectional
3. LSTM + CRF + chars embeddings + Static Bidirectional + Bahdanau Attention

#### 1.Concat-wise model
```text
	     precision    recall  f1-score   support

      EVENT       0.95      0.96      0.96      9443
       TIME       0.99      0.99      0.99    230666
        ORG       0.97      0.97      0.97     26911
        ART       0.97      0.98      0.98      3480
        FAC       0.94      0.89      0.92      7340
          O       0.88      0.97      0.92      3460
        LAW       0.89      0.91      0.90      2880
        LOC       0.95      0.94      0.95     11500
       NORP       0.86      0.97      0.91      1700
        PRN       0.96      0.96      0.96      2100
        DOC       0.80      0.80      0.80        20
          I       0.87      1.00      0.93        20

avg / total       0.98      0.98      0.98    299520

	     precision    recall  f1-score   support

    PISAHAN       0.98      0.99      0.98    169539
   KPENEGAS       0.98      0.98      0.98      2159
       NAFI       0.99      0.92      0.95     52116
         KP       0.99      1.00      0.99     23392
       NORP       0.99      1.00      1.00      7474
         GN       0.72      0.97      0.82      4080
         KN       0.95      0.99      0.97     13200
       ARAH       0.99      0.99      0.99     12860
        KPA       1.00      1.00      1.00      5756
       SUKU       0.99      0.39      0.56       220
         NO       1.00      0.99      0.99      2880
         KH       0.97      0.86      0.91       140
         KS       0.99      0.98      0.99      2064
          O       1.00      0.99      0.99      1060
        PAD       0.98      0.98      0.98      1160
 KETERANGAN       0.95      0.95      0.95       100
         KM       0.98      0.96      0.97       600
         KJ       1.00      0.99      0.99       220
         KB       0.80      0.93      0.86       380
         KT       1.00      0.65      0.79        20
   KPENGUAT       0.94      0.95      0.94        80
    KPEMERI       0.94      0.85      0.89        20

avg / total       0.98      0.98      0.98    299520


#### 2.Character-wise model
```text
	     precision    recall  f1-score   support

       TIME       0.74      0.82      0.78       474
        ART       0.95      0.96      0.96     11530
          I       0.86      0.80      0.83      1347
        LOC       0.88      0.90      0.89       174
        ORG       0.53      0.72      0.61       367
        PRN       0.65      0.80      0.72       173
        PAD       0.70      0.55      0.61       144
       NORP       0.72      0.60      0.65       575
        DOC       0.77      0.71      0.74        85
          O       0.86      0.51      0.64       105
      EVENT       0.00      0.00      0.00         1
        LAW       1.00      1.00      1.00         1

avg / total       0.91      0.91      0.91     14976


	     precision    recall  f1-score   support

         GN       0.91      0.96      0.94      8478
         KJ       0.96      0.95      0.96       109
         NO       0.97      0.71      0.82      2606
       NORP       0.95      0.99      0.97      1170
         KP       0.98      0.99      0.99       374
          O       0.53      0.68      0.59       204
       ARAH       0.84      0.99      0.91       660
    KPEMERI       0.98      0.97      0.98       641
         KB       0.98      0.99      0.98       287
    PISAHAN       0.00      0.00      0.00        11
       NAFI       1.00      0.98      0.99       144
         KA       0.88      1.00      0.93         7
         KS       0.93      0.93      0.93       103
         KH       1.00      1.00      1.00        53
   KPENGUAT       0.95      0.98      0.97        58
   KPENEGAS       0.71      1.00      0.83         5
         KT       0.97      0.93      0.95        30
        PAD       1.00      1.00      1.00        11
         KN       0.83      0.79      0.81        19
         KM       0.00      0.00      0.00         1
 KETERANGAN       1.00      0.50      0.67         4
       SUKU       0.33      1.00      0.50         1

avg / total       0.92      0.92      0.92     14976
```

#### 3.attention model
```text
	     precision    recall  f1-score   support

        PAD       0.97      1.00      0.98      9443
        FAC       1.00      1.00      1.00    230686
       TIME       1.00      0.99      1.00     26911
        PRN       1.00      1.00      1.00      3480
        ORG       1.00      0.99      0.99      7340
        LOC       0.99      1.00      0.99      3460
        LAW       1.00      0.99      0.99      2880
        DOC       0.99      1.00      0.99     11500
          O       1.00      1.00      1.00      1700
      EVENT       1.00      1.00      1.00      2100
        ART       1.00      1.00      1.00        20

avg / total       1.00      1.00      1.00    299520


	     precision    recall  f1-score   support

         KB       0.99      1.00      1.00    169539
         KT       1.00      1.00      1.00      2159
       ARAH       1.00      0.98      0.99     52116
   KPENEGAS       1.00      1.00      1.00     23392
         KJ       1.00      1.00      1.00      7474
        KPA       0.97      1.00      0.98      4080
         GN       1.00      1.00      1.00     13200
       SUKU       1.00      1.00      1.00     12860
         NO       1.00      1.00      1.00      5756
 KETERANGAN       1.00      0.95      0.97       220
         KA       1.00      1.00      1.00      2880
         KM       1.00      1.00      1.00       140
        PAD       1.00      1.00      1.00      2064
    KPEMERI       1.00      1.00      1.00      1060
       NAFI       1.00      0.99      1.00      1160
         KS       1.00      1.00      1.00       100
         KP       0.99      1.00      1.00       600
          O       1.00      0.97      0.99       220
         KN       0.95      1.00      0.97       380
   KPENGUAT       1.00      0.50      0.67        20
         KH       0.99      1.00      0.99        80
    PISAHAN       1.00      1.00      1.00        20

avg / total       1.00      1.00      1.00    299520
```
