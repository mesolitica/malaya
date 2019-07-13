option = {
    xAxis: {
        type: 'category',
        axisLabel: {
            interval: 0,
            rotate: 30
        },
        data: ['BERT-Bahasa SMALL (184MB)',
        'BERT-Bahasa BASE (467MB)',
        'BERT Multilanguage (714MB)',
        'XLNET-Bahasa 8-July-2019 (878MB)',
        'XLNET-Bahasa 9-July-2019 (878MB)']
    },
    yAxis: {
        type: 'value',
        min:0,
        max:1
    },
    grid:{
      bottom: 120
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [0.931452, 0.939288, 0.948821, 0.910186, 0.883091],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
