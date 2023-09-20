import os
import sys
from typing import List, Dict

from tqdm import tqdm
import numpy as np
import pandas as pd

try:
    path = os.path.abspath('../models/')
    sys.path.append(path)
    from text_classification import predict_zero_shot
except Exception as e:
    print(e)


def input_data(path: str) -> pd.DataFrame:
    """
    Преобразование входного файла в датафрейм

    Args:
        path (str): Путь до входного файла

    Returns:
        pd.DataFrame : Входная дата в виде датафрейма
    """
    return pd.read_csv(path)


def handle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обработка входного датафрейма:
    1. Удаление ненужных стоблцов
    2. удаление явных дубликатов
    3. Удаление ненужных метаданных

    Args:
        df (pd.DataFrame): Входной датафрейм

    Returns:
        pd.DataFrame: Датафрейм, готовый для классификации
    """
    # Удаление ненужных постов
    df = df[['text']]

    # Удаление явных дубликатов
    df = df.drop_duplicates(keep=False, ignore_index=True)

    # Удаление ненужных метаданных
    df = df.replace(to_replace=r"(?<=\[).+?(?=\])", value="", regex=True)
    df = df.replace(to_replace=r"\([^)]*\)", value="", regex=True)
    df = df.replace(to_replace=r"(@)\w+", value="", regex=True)
    df = df.replace(
        to_replace=r"^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$", value="", regex=True)

    return df


def classify_data(df: pd.DataFrame, post_themes: List[str]) -> pd.DataFrame:
    """
    Классификация всех постов по заданным тематикам

    Args:
        df (pd.DataFrame): Датафрейм с постами
        post_themes (List[str]): Темы постов

    Returns:
        pd.DataFrame: Датафрейм с номым добавленным полем: 'Label'
    """
    df['Label'] = np.nan
    for indx, paper in enumerate(tqdm(df['text'])):
        try:
            proba = predict_zero_shot(paper, post_themes)
            df['Label'].iloc[indx] = post_themes[np.argmax(proba)]
        except:
            label = 'Не удалось классифицировать'
            df['Label'].iloc[indx] = label

    return df


def fill_storage(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Наполнение хранилища уникальными и насыщенными постами

    Args:
        df (pd.DataFrame): Размеченный датафрейм

    Returns:
        Dict[str, List[str]]: Ключ - тема поста, значения - тексты постов
    """

    # Создадим хранилище
    storage: Dict[str, List[str]] = {}


def compare_with_other_posts(storage: Dict[str, List[str]], post: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Сверяем введенный пост на степень сходства с 10 постами в хранилище
    Если степень сходства > 0.9 с записью из хранилища,
    то выбирается более длинная запись, так как она более насыщена.
    Если степень сходства > 0.9 с несколькими записями их хранилища,
    то выбирается с максимальной степенью сходства

    Args:
        storage (Dict[str, pd.DataFrame]): Хранилище постов. Ключ - тема поста, значение - тексты постов
        post (Dict[str, str]):

    Returns:
        Dict[str, pd.DataFrame]: Измененное/неизмененное хранилище
    """


def out_put_data(storage: Dict[str, pd.DataFrame]) -> None:
    """
    Вывод 10 самых уникальных и насыщенных постов
    по всем тематикам

    Args:
        storage (Dict[str, pd.DataFrame]): Хранилище постов. Ключ - тема поста, значение - тексты постов
    Rerurns:
        None
    """
    pass


def main():
    # data_path = os.path.join('data', 'final_data.csv.gz')
    data_path = r'D:\projects\ai_news\data\test.csv'
    # post_themes = ['Блоги', 'Новости и СМИ', 'Развлечения и юмор', 'Технологии',
    #                'Экономика', 'Бизнес и стартапы', 'Криптовалюты', 'Путешествия',
    #                'Маркетинг, PR, реклама', 'Психология', 'Дизайн', 'Политика',
    #                'Искусство', 'Право', 'Образование и познавательное', 'Спорт',
    #                'Мода и красота', 'Здоровье и медицина', 'Картинки и фото',
    #                'Софт и приложения', 'Видео и фильмы', 'Музыка', 'Игры', 'Цитаты'
    #                'Еда и кулинария', 'Рукоделие', 'Финансы', 'Шоубиз', 'Другое']
    post_themes = ['Финансы', 'Технологии', 'Политика',
                   'Шоубиз', 'Fashion', 'Крипта', 'Путешествия/релокация',
                   'Образовательный контент', 'Развлечения', 'Общее']

    data = input_data(data_path)
    print(data.iloc[100])
    data = handle_data(data)
    classified_data = classify_data(data, post_themes)
    print(classified_data)
    # storage = fill_storage(classified_data)
    # out_put_data(storage)


if __name__ == "__main__":
    main()
