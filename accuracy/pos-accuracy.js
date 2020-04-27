option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-base (467.467MB)',
            'tiny-bert (57.7MB)',
            'albert-base (48.6MB)',
            'albert-tiny (22.4MB)',
            'xlnet-base (446.6MB)',
            'alxlnet-base (46.8MB)']
    },
    yAxis: {
        type: 'value',
        min: 0.93,
        max: 0.96
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.95254, 0.95343, 0.95105,
            0.93335, 0.95410, 0.95109],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
