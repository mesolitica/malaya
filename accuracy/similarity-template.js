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
        min: 0.85,
        max: 0.92
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.91236, 0.90235, 0.90219, 0.86835,
            0.85663, 0.91000],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
