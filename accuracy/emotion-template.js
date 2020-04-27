option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['multinomial (8MB)',
            'bert-base (425.6MB)',
            'tiny-bert (57.4MB)',
            'albert-base (48.6MB)',
            'albert-tiny (22.4MB)',
            'xlnet-base (446.MB)',
            'alxlnet-base (46.8MB)']
    },
    yAxis: {
        type: 'value',
        min: 0.9,
        max: 1.0
    },
    grid: {
        bottom: 100
    },
    backgroundColor: 'rgb(252,252,252)',
    series: [{
        data: [0.90371, 0.99790, 0.99701,
            0.99758, 0.99346, 0.99773, 0.99691],
        type: 'bar',
        label: {
            normal: {
                show: true,
                position: 'top'
            }
        },
    }]
};
