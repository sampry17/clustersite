window.addEventListener('load', function () {
    const selectedAlgorithm = localStorage.getItem('selectedAlgorithm');
    if (selectedAlgorithm) {
        const algorithmSelect = document.getElementById('algorithm-select');
        algorithmSelect.value = selectedAlgorithm;
        selectAlgorithm();
    }
});

function selectAlgorithm() {
    const algorithmSelect = document.getElementById('algorithm-select');
    const selectedAlgorithm = algorithmSelect.value;

    localStorage.setItem('selectedAlgorithm', selectedAlgorithm);

    const algorithmTemplates = document.getElementsByClassName('algorithm-template');
    for (let i = 0; i < algorithmTemplates.length; i++) {
        if (algorithmTemplates[i].id === selectedAlgorithm + '-template') {
            algorithmTemplates[i].style.display = 'block';
        } else {
            algorithmTemplates[i].style.display = 'none';
        }
    }
}

function showPlot() {
    const plotBlock = document.getElementById('plot-block');
    plotBlock.style.display = 'block';
}

function updateImage(data) {
    const img = document.getElementById('plot');
    img.src = 'data:image/png;base64,' + data;
    const plotDataInput = document.getElementById('plot_data');
    plotDataInput.value = data;
}

const plotData = '{{ plot_data|default:'' }}';
if (plotData) {
    updateImage(plotData);
}