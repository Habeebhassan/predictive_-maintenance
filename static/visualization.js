// function updateCharts(gasData, vibrationData) {
//     var ctxGas = document.getElementById('gasChart').getContext('2d');
//     var ctxVibration = document.getElementById('vibrationChart').getContext('2d');

//     var labels = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']; 

//     // Assuming gasData and vibrationData are arrays of values
//     var dataGas = gasData;  
//     var dataVibration = vibrationData;  

//     var gasChart = new Chart(ctxGas, {
//         type: 'bar',
//         data: {
//             labels: labels,
//             datasets: [{
//                 label: 'Gas Concentrations',
//                 data: dataGas,
//                 backgroundColor: 'rgba(75, 192, 192, 0.2)',
//                 borderColor: 'rgba(75, 192, 192, 1)',
//                 borderWidth: 1
//             }]
//         },
//         options: {
//             scales: {
//                 y: {
//                     beginAtZero: true
                
//                 }
//             }
//         }
//     });

//     var vibrationChart = new Chart(ctxVibration, {
//         type: 'line',
//         data: {
//             labels: labels,
//             datasets: [{
//                 label: 'Vibration Data',
//                 data: dataVibration,
//                 fill: false,
//                 borderColor: 'rgba(75, 192, 192, 1)',
//                 borderWidth: 1
//             }]
//         },
//         options: {
//             scales: {
//                 y: {
//                     beginAtZero: true
//                 }
//             }
//         }
//     });
// }

function updateCharts(gasData, vibrationData, regressionPredictions) {
    var ctxGas = document.getElementById('gasChart').getContext('2d');
    var ctxVibration = document.getElementById('vibrationChart').getContext('2d');

    var labelsGas = ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'];
    var labelsVibration = ['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3'];

    // Assuming gasData and vibrationData are arrays of values
    var dataGas = gasData;
    var dataVibration = vibrationData;

    var regressionValues = regressionPredictions.map(function (value) {
        return value[0];
    });

    var gasChart = new Chart(ctxGas, {
        type: 'bar',
        data: {
            labels: labelsGas,
            datasets: [{
                label: 'Gas Concentrations',
                data: dataGas,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    var vibrationChart = new Chart(ctxVibration, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Regression Predictions',
                data: regressionValues,
                pointBackgroundColor: 'blue',
                pointRadius: 5
            }]
        },
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    min: -15,
                    max: 1
                }
            }
        }
    });
}


function createScatterPlot(data) {
    var ctx = document.getElementById('scatterPlot').getContext('2d');

    var scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Anomalies',
                data: data.map((value, index) => ({x: index, y: value})), // Map data to x, y format
                pointBackgroundColor: 'red',
                pointRadius: 5
            }]
        },
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                },
                y: {
                    min: -15,
                    max: 1
                }
            }
        }
    });
}


// function ScatterPlot(data, labels) {
//     var ctx = document.getElementById('scatterPlot').getContext('2d');

//     var scatterChart = new Chart(ctx, {
//         type: 'scatter',
//         data: {
//             datasets: [{
//                 label: 'Anomalies',
//                 data: data.map((value, index) => ({x: index, y: value})), // Map data to x, y format
//                 pointBackgroundColor: 'red',
//                 pointRadius: 5
//             }]
//         },
//         options: {
//             scales: {
//                 x: {
//                     type: 'linear',
//                     position: 'bottom'
//                 },
//                 y: {
//                     min: -15,
//                     max: 1
//                 }
//             },
//             plugins: {
//                 tooltip: {
//                     callbacks: {
//                         label: function(context) {
//                             var label = labels[context.dataIndex]; // Get the label for the current data point
//                             return label;
//                         }
//                     }
//                 }
//             }
//         }
//     });
// }
