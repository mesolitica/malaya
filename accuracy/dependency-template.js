option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-base (467MB)',
        'xlnet-base (231MB)', 'albert-base (43MB)']
    },
    yAxis: {
        type: 'value',
        min:0.98,
        max:1.0
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.99491, 0.99636, 0.99023],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
