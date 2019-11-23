option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['multinomial-tfidf (73MB)',
        'BERT-Bahasa-base (467MB)',
        'XLNET-Bahasa-base (231MB)',
        'ALBERT-Bahasa-base (43MB)',
        'ALBERT-Bahasa-large (67.5MB)']
    },
    yAxis: {
        type: 'value',
        min:0.70,
        max:0.9
    },
    grid:{
      bottom: 120
    },
    title: {
        left: 'center',
        text: 'Emotion accuracy (f1 score 20% test)',
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.77725, 0.86970, 0.87029, 0.86506, 0.86866],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
