from herpetologist import check_type


def describe():
    """
    Describe kesalahan tatabahasa supported. 
    Full description at https://tatabahasabm.tripod.com/tata/salahtata.htm
    """
    d = [
        {'class': 0, 'Description': 'PAD', 'salah': '', 'betul': ''},
        {
            'class': 1,
            'Description': 'kesambungan subwords',
            'salah': '',
            'betul': '',
        },
        {
            'class': 2,
            'Description': 'tiada kesalahan',
            'salah': '',
            'betul': '',
        },
        {
            'class': 3,
            'Description': 'kesalahan frasa nama, Perkara yang diterangkan mesti mendahului "penerang"',
            'salah': 'Cili sos',
            'betul': 'sos cili',
        },
        {
            'class': 4,
            'Description': 'kesalahan kata jamak',
            'salah': 'mereka-mereka',
            'betul': 'mereka',
        },
        {
            'class': 5,
            'Description': 'kesalahan kata penguat',
            'salah': 'sangat tinggi sekali',
            'betul': 'sangat tinggi',
        },
        {
            'class': 6,
            'Description': 'kata adjektif dan imbuhan "ter" tanpa penguat.',
            'salah': 'Sani mendapat markah yang tertinggi sekali.',
            'betul': 'Sani mendapat markah yang tertinggi.',
        },
        {
            'class': 7,
            'Description': 'kesalahan kata hubung',
            'salah': 'Sally sedang membaca bila saya tiba di rumahnya.',
            'betul': 'Sally sedang membaca apabila saya tiba di rumahnya.',
        },
        {
            'class': 8,
            'Description': 'kesalahan kata bilangan',
            'salah': 'Beribu peniaga tidak membayar cukai pendapatan.',
            'betul': 'Beribu-ribu peniaga tidak membayar cukai pendapatan',
        },
        {
            'class': 9,
            'Description': 'kesalahan kata sendi',
            'salah': 'Umar telah berpindah daripada sekolah ini bulan lalu.',
            'betul': 'Umar telah berpindah dari sekolah ini bulan lalu.',
        },
        {
            'class': 10,
            'Description': 'kesalahan penjodoh bilangan',
            'salah': 'Setiap orang pelajar',
            'betul': 'Setiap pelajar.',
        },
        {
            'class': 11,
            'Description': 'kesalahan kata ganti diri',
            'salah': 'Pencuri itu telah ditangkap. Beliau dibawa ke balai polis.',
            'betul': 'Pencuri itu telah ditangkap. Dia dibawa ke balai polis.',
        },
        {
            'class': 12,
            'Description': 'kesalahan ayat pasif',
            'salah': 'Cerpen itu telah dikarang oleh saya.',
            'betul': 'Cerpen itu telah saya karang.',
        },
        {
            'class': 13,
            'Description': 'kesalahan kata tanya',
            'salah': 'Kamu berasal dari manakah ?',
            'betul': 'Kamu berasal dari mana ?',
        },
        {
            'class': 14,
            'Description': 'kesalahan tanda baca',
            'salah': 'Kamu berasal dari manakah .',
            'betul': 'Kamu berasal dari mana ?',
        },
        {
            'class': 15,
            'Description': 'kesalahan kata kerja tak transitif',
            'salah': 'Dia kata kepada saya',
            'betul': 'Dia berkata kepada saya',
        },
        {
            'class': 16,
            'Description': 'kesalahan kata kerja tak transitif',
            'salah': 'Dia kata kepada saya',
            'betul': 'Dia berkata kepada saya',
        },
        {
            'class': 17,
            'Description': 'kesalahan kata kerja transitif',
            'salah': 'Dia suka baca buku',
            'betul': 'Dia suka membaca buku',
        },
        {
            'class': 18,
            'Description': 'penggunaan kata yang tidak tepat',
            'salah': 'Tembuk Besar negeri Cina dibina oleh Shih Huang Ti.',
            'betul': 'Tembok Besar negeri Cina dibina oleh Shih Huang Ti',
        },
        {
            'class': 19,
            'Description': 'kesalahan frasa kerja tak transitif',
            'salah': 'berdasarkan pada keterangan ini',
            'betul': 'berdasarkan keterangan ini',
        },
        {
            'class': 20,
            'Description': 'kesalahan frasa kerja transitif',
            'salah': 'Dia membeli banyak buah',
            'betul': 'Dia banyak membeli buah',
        },
        {
            'class': 21,
            'Description': 'kesalahan frasa kerja pasif',
            'salah': 'Surat itu saga akan balas',
            'betul': 'Surat itu akan saga balas',
        },
    ]

    from malaya.function import describe_availability

    return describe_availability(d, transpose = False)
