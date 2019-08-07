option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-multilanguage','bert-base']
    },
    yAxis: {
        type: 'value',
        min:0.84,
        max:0.88
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.85, 0.87],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
