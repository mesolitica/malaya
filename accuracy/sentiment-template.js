option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','BERT','bidirectional','entity-network',
        'fast-text','fast-text-char','hierarchical','luong',
        'multinomial','xgb']
    },
    yAxis: {
        type: 'value',
        min:0.65,
        max:0.75
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.67,0.69,0.67,0.71,0.73,0.71,0.67,0.66,0.73,0.69],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
