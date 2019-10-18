option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['multinomial', 'bert-base (467MB)',
        'bert-small (185MB)', 'xlnet-base (231MB)', 'albert-base (43MB)']
    },
    yAxis: {
        type: 'value',
        min:0.4,
        max:0.73
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.41738, 0.70869, 0.72383, 0.72397, 0.69745],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
