option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','fast-text-char', 'luong', 'multinomial',
        'self-attention', 'xgboost', 'BERT']
    },
    yAxis: {
        type: 'value',
        min:0.75,
        max:0.89
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.86, 0.82, 0.86, 0.76, 0.83, 0.82, 0.88],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
