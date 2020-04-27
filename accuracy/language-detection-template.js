option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['fast-text (29.6MB)', 'deep-learning (48.8MB)']
    },
    yAxis: {
        type: 'value',
        min: 0.95,
        max: 1.0
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.97002, 0.97779],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
