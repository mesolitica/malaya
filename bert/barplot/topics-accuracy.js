option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['BERT-Bahasa pretraining',
        'BERT-Bahasa-subjective',
        'BERT-Bahasa sentiment',
        'BERT-Bahasa-emotion',
        'BERT-Bahasa-POS',
        'BERT-Bahasa-Entity',
        'BERT Multilanguage']
    },
    yAxis: {
        type: 'value',
        min:0.948,
        max:0.957
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.953462,
        0.949731,
        0.952000, 0.949808, 0.950846,
        0.949000,
        0.956846],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
