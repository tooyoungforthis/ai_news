# ai_news
ИИ-микросервис, который будет удалять дубликаты новостей
из фиксированного набора и классифицировать новости по заранее
известным категориям. В случае идентичной тематики в дублях
будет выбираться та, которая более насыщена целевой информацией.


Сперва обучаем алгоритм на данных для классификации,
которые нам даны от организатора.

Затем добавляем новость во временное хранилище
определяем ее класс, сверяем с исходным (если есть метка).

Смотрим на ее насыщенность, если она более насыщена или уникальная,
то оставляем ее и/или заменяем менее насыщенную.

Самые насыщенные статьи относительного обучающего набора
добавляем в другое временное хранилище
и используем для доработки(итеративного обучения) модели.

Затем выводим все насыщенные статьи по всем тематикам,
освобождаем временное хранилище
