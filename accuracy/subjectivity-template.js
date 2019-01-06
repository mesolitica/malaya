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
        min:0.81,
        max:0.9
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.83,0.84,0.85,0.88,0.89,0.88,0.84,0.82,0.89,0.85],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
