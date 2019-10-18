option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['bert-base (467MB)',
        'bert-small (185MB)', 'xlnet-base (231MB)', 'albert-base (43MB)']
    },
    yAxis: {
        type: 'value',
        min:0.945,
        max:0.96
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.95174, 0.95006, 0.95581, 0.95280],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
