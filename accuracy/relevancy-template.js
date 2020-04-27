option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-base (425.6MB)',
            'tiny-bert (57.4MB)',
            'albert-base (48.6MB)',
            'albert-tiny (22.4MB)',
            'xlnet-base (446.5MB)',
            'alxlnet-base (46.8MB)']
    },
    yAxis: {
        type: 'value',
        min: 0.65,
        max: 0.90
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.87264, 0.65601, 0.87183,
            0.84397, 0.88559, 0.87438],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
