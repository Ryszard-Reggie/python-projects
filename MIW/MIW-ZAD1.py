import config as cnf    # Zainportowany plik konfiguracyjny

import numpy as np
import pandas as pd
import os
import sys
import csv
from cerberus import Validator

# Treść zadania: =======================================================================================================

"""
    ZAD1

    1. Pobierz z repozytorium UCI następujące zbiory danych:
        a. Australian
        b. Credit Approval
        c. Breast Cancer Wisconsin

    2. Zaproponuj sposób przechowywania danych w programie (skorzystaj z wbudowanych typów danych lub stwórz własne).
       Zwróć uwagę, na rodzaj danych (numeryczne, symboliczne).
       
    3. Napisz program, który w oparciu o zaproponowane rozwiązanie z punktu 2,
       wczyta dane z pobranych wcześniej zbiorów danych.

    4. Stwórz mechanizm normalizujący dane.

    5. Zaimplementuj kilka sposobów (minimum 3) zapisu wczytanych danych,
       które będą różniły się od formatu, w którym aktualnie znajduje się repozytorium.

    6. Na podstawie zebranych doświadczeń stwórz plik konfiguracyjny do wymienionych wcześniej zbiorów danych.
       Plik powinien zawierać wszystkie ustawienia i parametry opisujące dane repozytorium,
       np. typ, liczba, zakres atrybutu, itp.
       Na podstawie pliku konfiguracyjnego program powinien sprawdzić poprawność wczytanych dany
       i poinformować użytkownika o ewentualnych  niezgodnościach.

    7. Dla chętnych. Spróbuj napisać mechanizm tworzący plik konfiguracyjny.
       System miałby wykrywać niektóre parametry zbioru danych, aby tworzenie pliku konfiguracyjnego było szybsze.
       
    1) ✓
        a) ✓
        b) ✓
        c) ✓
    2) ✓
    3) ✓    
    4) ✓    
    5) ✓
    6) ✓
    7) X
    
"""

# UŻYTECZNE FUNKCJE: ===================================================================================================


def display_list(list):
    for element in list:
        print(f'* {element}')


def display_dictionary(dictionary):
    for key, value in dictionary.items():
        print(f'Klucz: {key} | Wartość: {value}')


def display_dictionary_dictionary(dictionary):
    for key1, key2 in dictionary.items():
        for value in key2:
            print(f'Klucz: {key1} | Podklucz: {value} | Wartość: {key2[value]}')


# WCZYTANIE PLIKU Z DANYMI: ============================================================================================


def path_existence(path):
    """
        Funkcja ma za zadanie sprawdzić czy podana ścieżka istnieje.

        :param: path_to_file:    ścieżka do folderu
        :return:                Zwara True jeśli warunki okażą się spełnione, False w przypadku niepowodzenia
    """

    path = str(path)  # Ścieżka do pliku

    if os.path.exists(path):
        # print(f'Ścieżka \'{path}\' istnieje.')

        return True

    else:
        # print(f'Ścieżka \'{path}\' nie istnieje.')

        return False


def available_files(path):

    """
        Funkcja znajduje pliki znajdujące się w folderze i umieszcza ja w liście

        :param path:    ściażka do folderu
        :return:        zwraca listę plików z danej lokalizacji
    """

    files_list = list()

    for index, file in enumerate(os.listdir(path)):

        if os.path.isfile(os.path.join(path, file)):
            files_list.append(file)
            # print(f'* {file}')

    return files_list


def delimiter(path):

    """
        Funkcja ma za zadanie określenie separatora danych w podanum pliku

        :param path:    ścieżka prowadząca do pliku
        :return:        zwraca separator
    """

    file_open = open(path, 'r')
    lines = file_open.readlines()

    sniffer = csv.Sniffer()

    delimiter_list = list()

    for index, line in enumerate(lines):
        dialect = sniffer.sniff(line)

        delimiter = dialect.delimiter

        delimiter_list.append(delimiter)

        # print(f'Linia: {index} | Separator: {delimiter}')

    delimiter_count = delimiter_list.count(delimiter)

    if delimiter_count == len(delimiter_list):
        return delimiter


def enter_a_dataset():

    print(f'\nPodaj ścieżkę gdzie jest przechowyany plik z zestawem danych.\n')
    path = input(f'Ścieżka: ')

    while not path_existence(path):

        print(f'Ścieżka \'{path}\' nie istnieje.\n')
        print(f'Podaj ścieżkę gdzie jest przechowyany plik z zestawem danych.\n')
        path = input(f'Ścieżka: ')

    else:

        files_list = available_files(path)

        print(f'\nDostępne pliki:')

        display_list(files_list)

        print(f'\nWprowadź nazwę pliku z listy powyżej.\n')
        file = input(f'\nNazwa pliku: ')

        while file not in files_list:
            print(f'\nNie ma pliku \'{file}\' na liście.\n')
            print(f'\nWprowadź nazwę pliku z listy powyżej.\n')
            file = input(f'\nNazwa pliku: ')

        else:
            path_with_file = os.path.join(path, file)
            separator = delimiter(path_with_file)

            return pd.read_csv(path_with_file, header=None, encoding='utf-8', sep=separator)


dataframe = enter_a_dataset()

# WALIDACJA: ===========================================================================================================


def records_validation(dataframe_records):
    records_validation_list = list()

    # validation = True

    validation_dict = dict()

    # validation_list = list()

    for dataset_name, schema in cnf.datasets_schemas.items():
        v = Validator(schema, allow_unknown=False, require_all=True, purge_unknown=True)

        for index, record in enumerate(dataframe_records):
            if v.validate(record):
                # print(f'Wiersz: {index} | Walidacja: {v.validate(record)}')

                records_validation_list.append(True)

            elif not v.validate(record):
                # print(f'Wiersz: {index} | Walidacja: {v.validate(record)}\nBłąd: {v.errors}')

                records_validation_list.append(False)

        trueCount = records_validation_list.count(True)  # Liczba zwalidowanych wierszy z Datasetu
        falseCount = records_validation_list.count(False)  # Liczba niezwalidowanych wierszy z Datasetu

        # print(f'* Ilość wierszy: {len(records_validation_list)}')
        # print(f'* Zwalidowanych: {trueCount}')
        # print(f'* Niezwalidowanych: {falseCount}')

        if falseCount > 0:
            validation_dict[dataset_name] = False

        elif falseCount == 0:
            validation_dict[dataset_name] = True

        records_validation_list.clear()

    return validation_dict


def datafame_validation(dataframe):
    print(f'\nWaliduję...\n')

    dataframe = dataframe.replace({'?': None})

    dataframe_records = dataframe.to_dict(orient='records')

    validation_dict = records_validation(dataframe_records)

    validation_list = list()

    for value in validation_dict.values():
        validation_list.append(value)

    if any(validation_list):
        for key, value in validation_dict.items():
            if value:
                print(f'\nWalidacja zakończona pomyślnie dla \'{key}\'\n')
                return key

    else:
        print(f'\nPodany plik nie przeszedł walidacji. Spróbuj z innym plikiem.\n')
        sys.exit()


dataset_name = datafame_validation(dataframe)

# INFORMACJE: ==========================================================================================================

# info = df.info()                              # Informacje o zestawie danych
# print(f'{info}\n')
#
# describe = df.describe()
# print(f'{describe}\n')

# rowsNumber = len(dataframe)                       # Liczba wierszy (rows) w ramce danych
# columnsNumber = len(dataframe.columns)                # Liczba kolumn (columns) w ramce danych
#
# pd.set_option('max_columns', rowsNumber)    # Ustawienie opcji wyświetlania wszystkich kolumnm
# pd.set_option('max_rows', columnsNumber)      # Ustawienie opcji wyświetlania wszyskich wierszy

# POMINIĘCIE KOLUMN: ===================================================================================================


def drop_columns(dataframe, drop_columns):
    dataframe = dataframe.drop(columns=drop_columns, axis='index')

    print(f'\nKolumny wyznaczone do pominięcia: {drop_columns}\n')

    return dataframe


dataframe = drop_columns(dataframe, cnf.datasets[dataset_name]['drop_columns'])

# print(f'\nPróbki dataset\'u po pominięciu kolumn: \n{dataframe.sample(5)}\n')

# WYKRYCIE I ZASTĄPIENIE BRAKUJĄCYCH WARTOŚCI: =========================================================================


def any_missing_values(dataframe, missing_values):
    for mark in range(len(missing_values)):
        dataframe = dataframe.replace(to_replace=missing_values[mark], value=np.NaN)

    sum_of_missing_values = dataframe.isnull().values.sum()
    print(f'\nLiczba brakujących wartości: {sum_of_missing_values}\n')

    fill_missing_values_options_list = ['mean', 'mode', 'drop']

    if sum_of_missing_values > 0:

        print(f'\nDostępne tryby zastąpienia brakujących wartości:')
        display_list(fill_missing_values_options_list)

        fill_missing_values_option = input(f'\nWprowadź nazwę trybu: ')

        while fill_missing_values_option not in fill_missing_values_options_list:
            print(f'\n\'{fill_missing_values_option}\' nie ma w opcjach.\n')

            fill_missing_values_option = input(f'\nWprowadź nazwę trybu: ')

        else:
            if fill_missing_values_option == 'mean':
                dataframe = dataframe.fillna(dataframe.mean())

            elif fill_missing_values_option == 'mode':
                for column in dataframe.columns:
                    dataframe[column] = dataframe[column].fillna(dataframe[column].value_counts().index[0])

            elif fill_missing_values_option == 'drop':
                dataframe = dataframe.dropna(axis='index')

        for column in dataframe.columns:
            if dataframe[column].dtype == 'object':
                dataframe[column] = dataframe[column].fillna(dataframe[column].value_counts().index[0])

    for column in dataframe.columns:
        dataframe = dataframe.astype({column: cnf.datasets[dataset_name]['columns_types'].get(column)})

    return dataframe


dataframe = any_missing_values(dataframe, cnf.datasets[dataset_name]['missing_values'])

# print(f'\nPróbki dataset\'u po zastępieniu brakujących wartości: \n{dataframe.sample(5)}\n')

# ZAMIANA WARTOŚCI SYMBOLICZNYCH NA WARTOŚCI NUMERYCZNE: ===============================================================


def mapping_column(dataframe, column):
    symbol_map = dict()
    number = 1

    for symbol in sorted(dataframe[column].unique()):
        symbol_map[symbol] = number
        number = number + 1

    dataframe[column] = dataframe[column].map(symbol_map)

    return dataframe


def mapping_columns(dataframe, columns):
    # print(f'Kolumny: {columns} | Typ danych: {type(columns)} | Długość: {len(columns)}\n')
    print(f'\nKolumny wyznaczone do zamiany wartości symbolicznych: {columns}\n')

    for column in range(len(columns)):
        mapping_column(dataframe, columns[column])

    return dataframe


dataframe = mapping_columns(dataframe, cnf.datasets[dataset_name]['non-numeric_columns'])    # To musi być lista

# print(f'\nPróbki dataset\'u po zamianie wartości symbolicznych: \n{dataframe.sample(5)}\n')

# NORMALIZACJA: =======================================================================================================


def input_range_from():
    range_from = input(f'\nOd: ')

    try:
        range_from = int(range_from)
        return range_from
    except:
        print(f'\nWprowadź liczbę!\n')
        input_range_from()


def input_range_to():
    range_to = input(f'\nDo: ')

    try:
        range_to = int(range_to)
        return range_to
    except:
        print(f'\nWprowadź liczbę!\n')
        input_range_to()


def data_normalization(dataframe, normalization_columns):
    print(f'\nKolumny wyznaczone do normalizacji: {normalization_columns}\n')

    print(f'\nWprowadź zakres normalizacji:')
    range_from = input_range_from()
    range_to = input_range_to()

    while range_from >= range_to:

        print(f'\nWartość \'OD\' ({range_from}) musi być mniejsza od wartości \'DO\' ({range_to}).\n')

        print(f'\nWprowadź zakres normalizacji:')
        range_from = input_range_from()
        range_to = input_range_to()

    else:

        for column in normalization_columns:

            value = dataframe[column]
            minimum = dataframe[column].min()
            maximum = dataframe[column].max()

            formula = (value - minimum) / (maximum - minimum)

            if range_from == 0 and range_to == 1:
                dataframe[column] = formula
            else:
                formula = formula * (range_to - range_from) + range_from
                dataframe[column] = formula

        return dataframe


dataframe = data_normalization(dataframe, cnf.datasets[dataset_name]['normalization_columns'])

# print(f'\nPróbki dataset\'u po normalizacji: \n{dataframe.sample(5)}\n')

# ZAPIS ZNORMALIZOWANYCH DANYCH ========================================================================================


def save_dataframe(dataframe):

    saveFormatList = ['csv', 'json', 'html', 'excel']

    print(f'\nDostępne formaty zapisu:')
    display_list(saveFormatList)

    saveFormat = input(f'\nWprowadź nazwę formatu: ')
    saveFormat = str(saveFormat)

    while saveFormat not in saveFormatList:
        print(f'\nWprowadzona nazwa \'{saveFormat}\' jest nie poprawna.\n')

        saveFormat = input(f'Wprowadź nazwe formatu: ')
        saveFormat = str(saveFormat)

    else:
        path = input(f'\nPodaj ścieżkę gdzie zapisać plik: ')
        path = str(path)

        while not os.path.exists(path):

            print(f'\nNieprawidłowa ścieżka!\n')

            path = input(f'\nPodaj ścieżkę gdzie zapisać plik: ')
            path = str(path)

        else:
            filename = input(f'\nPodaj nazwe pliku: ')
            filename = str(filename)

        savePath = path + filename

        if saveFormat == 'csv' and saveFormat in saveFormatList:

            separator = input(f'\nPodaj separator: ')
            separator = str(separator)

            dataframe.to_csv(path_or_buf=r'{}.csv'.format(savePath), sep=separator, header=False, index=False)

        elif saveFormat == 'json' and saveFormat in saveFormatList:

            dataframe.to_json(path_or_buf=r'{}.json'.format(savePath))

        elif saveFormat == 'html' and saveFormat in saveFormatList:

            html = dataframe.to_html()

            textFile = open(savePath + '.html', 'w')
            textFile.write(html)
            textFile.close()

        elif saveFormat == 'excel' and saveFormat in saveFormatList:

            sheetName = input(f'\nPodaj nazwe arkusza: ')
            sheetName = str(sheetName)

            dataframe.to_excel(excel_writer=savePath + '.xlsx', sheet_name=sheetName, index=False, header=False)


save_dataframe(dataframe)
