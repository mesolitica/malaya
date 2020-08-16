option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-base (470MB)', 'tiny-bert (125MB)',
            'albert-base (180MB)', 'tiny-albert (56.7MB)',
            'xlnet-base (498MB)']
    },
    yAxis: {
        type: 'value',
        min: 69,
        max: 82
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [80.35, 74.89, 79.01, 70.84, 81.43],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
