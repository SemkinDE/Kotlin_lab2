package smile

import smile.regression.GradientTreeBoost
import smile.data.formula.Formula
import smile.io.Read
import smile.validation.CrossValidation
import org.apache.commons.csv.CSVFormat

fun main() {
    // Чтение данных из CSV
    val dsFileFormat = CSVFormat.DEFAULT.builder()
        .setHeader()
        .setSkipHeaderRecord(true)
        .setDelimiter(',')
        .build()

    // Чтение данных
    val dataset = Read.csv("src/main/resources/prostate.csv", dsFileFormat)

    // Выводим структуру данных для проверки
    println(dataset)

    // Формула для предсказания
    val formula = Formula.lhs("lpsa")

    // Кросс-валидация для регрессии
    val res = CrossValidation.regression(
        10, formula, dataset,
        { formula, data -> GradientTreeBoost.fit(formula, data) }
    )

    println(res)
}

