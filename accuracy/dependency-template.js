option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-base (426MB)',
            'tiny-bert (59.5MB)',
            'albert-base (50MB)',
            'albert-tiny (24.8MB)',
            'xlnet-base (450.2MB)',
            'alxlnet-base (50.0MB)']
    },
    yAxis: {
        type: 'value',
        min: 0.96,
        max: 1.0
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.99562, 0.98102, 0.98785, 0.96997,
            0.99678, 0.99475],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
