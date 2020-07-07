option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['transformer-small (43MB)',
            'transformer-base (234MB)', 'transformer-large (815MB)']
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
        data: [0.692, 0.792, 0.714],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
