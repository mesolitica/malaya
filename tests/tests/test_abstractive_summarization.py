import os
import sys
import malaya
import logging
import re

logging.basicConfig(level=logging.DEBUG)


string = """
PELETAKAN jawatan Tun Dr Mahathir Mohamad sebagai Pengerusi Parti Pribumi Bersatu Malaysia (Bersatu) ditolak di dalam mesyuarat khas Majlis Pimpinan Tertinggi (MPT) pada 24 Februari lalu.

Justeru, tidak timbul soal peletakan jawatan itu sah atau tidak kerana ia sudah pun diputuskan pada peringkat parti yang dipersetujui semua termasuk Presiden, Tan Sri Muhyiddin Yassin.

Bekas Setiausaha Agung Bersatu Datuk Marzuki Yahya berkata, pada mesyuarat itu MPT sebulat suara menolak peletakan jawatan Dr Mahathir.

"Jadi ini agak berlawanan dengan keputusan yang kita sudah buat. Saya tak faham bagaimana Jabatan Pendaftar Pertubuhan Malaysia (JPPM) kata peletakan jawatan itu sah sedangkan kita sudah buat keputusan di dalam mesyuarat, bukan seorang dua yang buat keputusan.

"Semua keputusan mesti dibuat melalui parti. Walau apa juga perbincangan dibuat di luar daripada keputusan mesyuarat, ini bukan keputusan parti.

"Apa locus standy yang ada pada Setiausaha Kerja untuk membawa perkara ini kepada JPPM. Seharusnya ia dibawa kepada Setiausaha Agung sebagai pentadbir kepada parti," katanya kepada Harian Metro.

Beliau mengulas laporan media tempatan hari ini mengenai pengesahan JPPM bahawa Dr Mahathir tidak lagi menjadi Pengerusi Bersatu berikutan peletakan jawatannya di tengah-tengah pergolakan politik pada akhir Februari adalah sah.

Laporan itu juga menyatakan, kedudukan Muhyiddin Yassin memangku jawatan itu juga sah.

Menurutnya, memang betul Dr Mahathir menghantar surat peletakan jawatan, tetapi ditolak oleh MPT.

"Fasal yang disebut itu terpakai sekiranya berhenti atau diberhentikan, tetapi ini mesyuarat sudah menolak," katanya.

Marzuki turut mempersoal kenyataan media yang dibuat beberapa pimpinan parti itu hari ini yang menyatakan sokongan kepada Perikatan Nasional.

"Kenyataan media bukanlah keputusan rasmi. Walaupun kita buat 1,000 kenyataan sekali pun ia tetap tidak merubah keputusan yang sudah dibuat di dalam mesyuarat. Kita catat di dalam minit apa yang berlaku di dalam mesyuarat," katanya.
"""


def test_huggingface():

    cache_dir = os.path.join(os.getcwd(), 'cache')
    print(cache_dir)

    models = malaya.summarization.abstractive.available_huggingface
    for m in models:
        print(m)
        model = malaya.summarization.abstractive.huggingface(model=m, cache_dir=cache_dir)
        print(model.generate([string], max_length=256))
        del model

    os.system(f'rm -f {cache_dir}')
