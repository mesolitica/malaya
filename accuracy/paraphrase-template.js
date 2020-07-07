option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['transformer-tiny (18.4MB)', 'transformer-small (43MB)',
            'transformer-base (234MB)', 'tiny-bert (60.6MB)', 'bert (449MB)']
    },
    yAxis: {
        type: 'value',
        min: 0.5,
        max: 0.8
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.594, 0.737, 0.792, 0.609, 0.696],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
