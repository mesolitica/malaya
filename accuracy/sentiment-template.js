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
        min:0.80,
        max:0.85
    },
    grid:{
      bottom: 100
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.80541, 0.84132, 0.84123, 0.83838, 0.81992],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
