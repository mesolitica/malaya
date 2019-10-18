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
        min:0.80,
        max:0.87
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.86030, 0.81741, 0.83809],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
