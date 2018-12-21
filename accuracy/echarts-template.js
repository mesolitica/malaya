option = {
    xAxis: {
        type: 'category',
        data: ['concat', 'bahdanau', 'luong', 'entity-network', 'crf','attention']
    },
    yAxis: {
        type: 'value',
        min:0.5,
        max:1.0
    },
    backgroundColor:'rgb(252,252,252)',
    series: [{
        data: [1.0,1.0,1.0,1.0, 0.987, 1.0],
        type: 'bar',
        label: {
                normal: {
                    show: true,
                    position: 'top'
                }
            },
    }]
};
