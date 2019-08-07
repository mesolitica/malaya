option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','luong', 'multinomial',
        'self-attention', 'xgboost', 'bert-multilanguage', 'bert-base',
        'bert-small']
    },
    yAxis: {
        type: 'value',
        min:0.76,
        max:0.93
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.80, 0.81, 0.89, 0.77, 0.85, 0.92,
        0.92,0.905],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
