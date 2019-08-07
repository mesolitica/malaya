option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau', 'luong', 'multinomial',
        'self-attention', 'xgb', 'bert-multilanguage',
        'bert-base', 'base-small']
    },
    yAxis: {
        type: 'value',
        min:0.77,
        max:0.805
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.79, 0.79, 0.79, 0.78, 0.8, 0.8,
        0.8, 0.80],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
