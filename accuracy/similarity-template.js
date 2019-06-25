option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','self-attention', 'dilated-cnn', 'BERT']
    },
    yAxis: {
        type: 'value',
        min:0.76,
        max:0.83
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.79, 0.77, 0.77, 0.83],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
