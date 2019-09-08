option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['concat', 'bahdanau', 'luong', 'bert-multilanguage',
        'bert-base', 'bert-small']
    },
    yAxis: {
        type: 'value',
        min:0.9,
        max:0.97
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.913031, 0.903527, 0.901504,
        0.966343, 0.964343, 0.963971],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
