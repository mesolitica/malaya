option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30,
        },
        data: ['sklearn-crfsuite (6MB)',
        'BERT-Bahasa-base (467MB)',
        'BERT-Bahasa-small (185MB)',
        'XLNET-Bahasa-base (231MB)',
        'ALBERT-Bahasa-base (43MB)',
        'ALXLNET-Bahasa-base (34MB)']
    },
    yAxis: {
        type: 'value',
        min:0.90,
        max:0.97
    },
    grid:{
      bottom: 120
    },
    title: {
        left: 'center',
        text: 'Part-Of-Speech accuracy (f1 score 20% test)'
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.91810, 0.95174, 0.95006, 0.95581, 0.95280, 0.95185],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
