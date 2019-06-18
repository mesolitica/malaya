option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['BERT-Bahasa pretraining',
        'BERT-Bahasa sentiment',
        'BERT-Bahasa-emotion', 
        'BERT-Bahasa-POS',
        'BERT-Bahasa-Entity',
        'BERT Multilanguage']
    },
    yAxis: {
        type: 'value',
        min:0.75,
        max:1.0
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.895326, 0.959744, 0.950210, 0.931954,
        0.945695, 0.948821],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
