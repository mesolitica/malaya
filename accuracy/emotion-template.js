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
        min:0.73,
        max:0.81
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.79,0.77,0.80,0.76,0.77,0.75,0.80,0.79,0.75,0.79],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
