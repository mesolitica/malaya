option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','luong', 'multinomial',
        'self-attention', 'xgboost', 'bert-multilanguage',
        'bert-base','bert-small']
    },
    yAxis: {
        type: 'value',
        min:0.71,
        max:0.89
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.85, 0.85, 0.72, 0.83, 0.82, 0.87, 0.87, 0.87],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
