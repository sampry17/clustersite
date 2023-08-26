$(document).ready(function() {
    // Обработчик события на изменение значения выбранного элемента
    $('#algorithm-select').change(function() {
        // Скрыть все HTML-шаблоны
        $('.algorithm-template').hide();

        // Получить выбранный алгоритм
        var selectedAlgorithm = $(this).val();

        // Отобразить соответствующий HTML-шаблон
        $('#' + selectedAlgorithm + '-template').show();
    });
});
