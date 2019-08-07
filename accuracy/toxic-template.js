option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bahdanau','luong', 'multinomial',
        'self-attention', 'bert-multilanguage', 'bert-base', 'bert-small']
    },
    yAxis: {
        type: 'value',
        min:0.55,
        max:0.73
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.68, 0.67, 0.56, 0.58, 0.715010, 0.724062, 0.700970],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
