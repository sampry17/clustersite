$(document).ready(function() {
    $(".solution-button").on("click", function() {
        var k = $(this).data("k");
        var n = $(this).data("n");
        var lowerBound = $(this).data("lower-bound");
        var upperBound = $(this).data("upper-bound");
        var metrika = $(this).data("metrika");
        var q1 = $(this).data("q1");
        var q2 = $(this).data("q2");
        var q3 = $(this).data("q3");
        var w = $(this).data("w");

        // Отправка данных на сервер через AJAX-запрос
        $.ajax({
            url: "/save-cluster/",  // URL-адрес обработчика на сервере
            type: "POST",
            data: {
                k: k,
                n: n,
                lower_bound: lowerBound,
                upper_bound: upperBound,
                metrika: metrika,
                q1: q1,
                q2: q2,
                q3: q3,
                w: w
            },
            success: function(response) {
                // Обработка успешного ответа сервера
                alert("Решение выбрано успешно!");
            },
            error: function(xhr, textStatus, errorThrown) {
                // Обработка ошибки запроса
                alert("Произошла ошибка при выборе решения.");
            }
        });
    });
});