// 初始化ECharts实例
const scatterChart = echarts.init(document.getElementById("chart"));

var data = [[10.0, 8.04], [10.0, -8.04], [-10.0, -8.04], [-10.0, 8.81], [11.0, 8.33]];
var option = {};
scatterOption = {
    title:{
        text: "City Display",
        x:'center',
        y:'top',
        textAlign:'left'
    },
    xAxis: {
        // min: -200,
        // max: 200,
        type: 'value',
        // show:false
    },
    yAxis: {
        // min: -200,
        // max: 200,
        type: 'value',
        // show:false
    },
    grid:{
        top: "10%",
        left: '10%',
        right: '10%',
        bottom: '10%',
        containLabel: true
    },
    animation: false,
    series: [{
        type: 'scatter',
        data: data
    },
    {
        type: 'line',
        data: [[10.0, 8.04], [10.0, -8.04], [-10.0, -8.04], [-10.0, 8.81], [11.0, 8.33]],
        connectNulls: true
      }    
    ]
};
scatterChart.setOption(scatterOption)
console.log("scatterChart Chart initialized.")