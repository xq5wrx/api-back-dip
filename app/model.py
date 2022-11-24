from fastapi import UploadFile
import func
import conn


def generateModel():
    # Получение и преобразование данных
    train_input, train_output = func.getTrainData()
    epochs = 50

    # Анализ данных методом главных компонент
    # pca_result = func.startPcaAnalysis(train_input)

    # Оценка
    # all_scores, val_mse, all_mae_histories = func.getGrade(train_input, train_output, epochs)

    # Вывод результата оценки
    # func.getGradePlots(all_scores, val_mse, all_mae_histories, epochs)

    # Получение и преобразование данных
    test_input, test_output, denorm = func.getMainData(train_input, train_output)

    # Построение модели
    test_mse_score, test_mae_score, predicted = func.makeFinalModel(test_input, test_output, epochs)

    # Получение результат
    func.getFinalResult(test_mae_score, test_mse_score, predicted, test_output, denorm)

    return 1


def importDataSql(file: UploadFile):
    # for save and open file
    # with open("input_files/" + file.filename, 'wb') as image:
    #     content = await file.read()
    #     image.write(content)
    #     image.close()
    #
    # with open("input_files/test.sql", "r") as f:
    #     sql = f.read()
    contents = file.file.read()
    sql = str(contents, 'utf-8')

    connection = conn.getConnection()
    cursor = connection.cursor()

    cursor.execute(sql)

    connection.close()

    return 1