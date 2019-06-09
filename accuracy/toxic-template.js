option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','fast-text-char', 'luong', 'multinomial',
        'self-attention', 'BERT']
    },
    yAxis: {
        type: 'value',
        min:0.70,
        max:0.80
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.75, 0.77, 0.75, 0.71, 0.76, 0.79],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
