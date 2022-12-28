import config as cnf
from cerberus import Validator

from tkinter import *
from tkinter import filedialog

import os
import math
import pandas as pd
import numpy as np
import csv
from operator import itemgetter

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ZADANIE: =============================================================================================================

"""
    Dataset:
    Dane:   Klasa decyzyjna:    Odległości:
    -----   0                   3
    -----   1                   1
    -----   1                   2
    -----   0                   2
    -----   0                   4
    -----   1                   2
    -----   0                   5

    I wariant - Z pośród wszystkich odległości wybiera k najmniejszych odległości

    Dla k = 5 (5 najbliższych sąsiadów)

    Klasa decyzyjna:    Odległości:
    1                   1
    1                   2
    0                   2
    1                   2
    0                   3

    Klasa:      Liczba wystąpień:
    0           2
    1           3

    Decyzja KNN: klasa 1

    Max k to: ilość próbek wzorcowych
    --------------------------------------------------------------------------------------------------------------------
    II wariant - Z każdej klasy decyzyjnej bierzemy po k najmniejszych wartości:

    Dla k = 2

    Klasa decyzyjna:    Odległości:
    0                   2
    0                   3

    Klasa decyzyjna:    Odległości:
    1                   1
    1                   2

    Klasa:      Liczba wystąpień:
    0           2 + 3 = 5
    1           1 + 2 = 3

    Decyzja KNN: klasa 1

    Max k to: liczebność najmniejszej liczby klasy decyzyjnej
    --------------------------------------------------------------------------------------------------------------------
    Napisz program, który sklasyfikuje jeden podany przez użytkownika obiekt (próbka testowa).
    Klasyfikację należy wybrać w dwóch wariantach na podstawie datasetów z poprzednich zajęć.
    a) Sprawdzenie czy podane k jest dozwolone
    b) mierzyć odległości za pomocą metryki między dwoma obiektami 

    Co jest potrzebne żeby sklasyfikować jeden obiekt (próbkę testową)?
    1) próbka testowa (obiekt) - lista
    2) próbki wzorcowe
    3) parametr k
    4) metryka
"""

# FUNKCJA WALIDACYJNA: -------------------------------------------------------------------------------------------------


def records_validation_function(dataframe_records):
    records_validation_list = list()

    validation = True

    validation_dict = dict()

    validation_list = list()

    for dataset_name, schema in cnf.normalized_datasets_schemas.items():
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


def new_row_validation_function(dataframe_records):
    new_row_validation_list = list()

    validation = True

    validation_dict = dict()

    validation_list = list()

    for dataset_name, schema in cnf.new_row_datasets_schemas.items():
        v = Validator(schema, allow_unknown=False, require_all=True, purge_unknown=True)

        for index, record in enumerate(dataframe_records):
            if v.validate(record):
                # print(f'Wiersz: {index} | Walidacja: {v.validate(record)}')

                new_row_validation_list.append(True)

            elif not v.validate(record):
                # print(f'Wiersz: {index} | Walidacja: {v.validate(record)}\nBłąd: {v.errors}')

                new_row_validation_list.append(False)

        trueCount = new_row_validation_list.count(True)  # Liczba zwalidowanych wierszy z Datasetu
        falseCount = new_row_validation_list.count(False)  # Liczba niezwalidowanych wierszy z Datasetu

        # print(f'* Ilość wierszy: {len(records_validation_list)}')
        # print(f'* Zwalidowanych: {trueCount}')
        # print(f'* Niezwalidowanych: {falseCount}')

        if falseCount > 0:
            validation_dict[dataset_name] = False

        elif falseCount == 0:
            validation_dict[dataset_name] = True

        new_row_validation_list.clear()

    return validation_dict


def datafame_validation(dataframe):
    print(f'\nWaliduję...\n')

    # dataframe = dataframe.replace({'?': None})

    dataframe_records = dataframe.to_dict(orient='records')

    validation_dict = records_validation_function(dataframe_records)

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


def new_row_validation(dataframe):
    print(f'\nWaliduję...\n')

    # dataframe = dataframe.replace({'?': None})

    new_row = dataframe.to_dict(orient='records')

    validation_dict = new_row_validation_function(new_row)

    validation_list = list()

    for value in validation_dict.values():
        validation_list.append(value)

    if any(validation_list):
        for key, value in validation_dict.items():
            if value:
                return key

    else:
        print(f'\nPodany plik nie przeszedł walidacji. Spróbuj z innym plikiem.\n')


# OKREŚLENIE SEPARATORA W PLIKU: =======================================================================================


def extracting_separator(path):
    """
        Funkcja ma za zadanie określenie separatora danych w podanum pliku.

        :param path:    ścieżka prowadząca do pliku
        :return:        zwraca separator
    """

    delimiter_detect = None

    file_open = open(path, 'r')
    lines = file_open.readlines()

    sniffer = csv.Sniffer()

    delimiter_list = list()

    for index, line in enumerate(lines):
        dialect = sniffer.sniff(line)

        delimiter_detect = dialect.delimiter

        delimiter_list.append(delimiter_detect)

        # print(f'Linia: {index} | Separator: {delimiter}')

    delimiter_count = delimiter_list.count(delimiter_detect)

    if delimiter_count == len(delimiter_list):
        return delimiter_detect


# METRYKI DO OBLICZANIA DYSTANSU: ======================================================================================


def euclidean(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        distance += math.pow(row1[i] - row2[i], 2)

    return math.sqrt(distance)


def manhattan(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i])

    return distance


def hamming(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i] / len(row1))

    return distance


def chebyshev(row1, row2):
    distance = 0.0

    for i in range(len(row1) - 1):
        if abs(row1[i] - row2[i]) > distance:
            distance = abs(row1[i] - row2[i])

    return distance

# FUNKCJE SŁUŻĄCE DO PREDYKCJI: ========================================================================================


def calculate_distances(dataset, sample, metrick_function):
    distances_list = list()

    # Dla każdego wiersza w datasecie ...
    for row in range(len(dataset)):
        # oblicz dystans między obecnym wierszem (`row`), a wybraną próbką (`sample`).
        distance = eval(metrick_function)(sample, dataset[row][:-1])
        # Dodanie do listy `distances` listy zawierającej dystans i wiersz.
        distances_list.append([distance, dataset[row]])

    # Posortowanie listy odległości:
    sorted_distances_list = sorted(distances_list, key=itemgetter(0))
    sorted_distances_list = sorted_distances_list[1:]

    # print(f'\nPosortowana lista odległości:\n{sorted_distances_list}\n')
    return sorted_distances_list


def k_nearest_neighbors(sorted_distances, k_value):
    """
        Funkcja zwraca liste najbliższych k sąsiadów.

        list_of_neighbors:   lista sąsiadów
        k_value:             wartość k
    """

    neighbors_list = list()         # Lista do przechowania k najbliższych sąsiadów.

    # Dla każdego sąsiada z zakresu od 0 do `k_value` ...
    for neighbor in range(k_value):
        # dodaj do listy sąsiadów wiersze sąsiadów z posortowanej listy dystansów (najmniejsze dystanse).
        neighbors_list.append(sorted_distances[neighbor + 1])

    # print(f'\nLista sąsiadów:\n{neighbors_list[0:2]}\n')
    return neighbors_list


def classes_function(neighbors_list):
    classes_dictionary = dict()     # Słownik na klasy

    # Dla każdego sąsiada z zakresu od 0 do długości listy `neighbors_list` ...
    for neighbor in range(len(neighbors_list)):
        response = neighbors_list[neighbor][-1][-1]     # Kolumna przechowująca decyzję
        # print(f"Klasyfikator: {response}")

        # Jeśli klasyfiaktor jest już w słowniku ...
        if response in classes_dictionary:
            # zwiększ wartość o 1.
            classes_dictionary[response] += 1
        # W przeciwnym razie ...
        else:
            # ustaw wartość na 1.
            classes_dictionary[response] = 1

    # Ilość powtórzeń maksymalnych klasyfikatorów
    repeat = 0

    # Najczęściej występujący klasyfikator:
    most_common_classifier = max(classes_dictionary, key=classes_dictionary.get)

    # Wartość największej liczby wystąpień klasyfikaotra:
    largest_number_of_classifiers = max(classes_dictionary.values())

    # Zwiększenie licznika powtórzeń największej liczby klasyfikatorów:
    for element in classes_dictionary.values():
        if element == largest_number_of_classifiers:
            repeat += 1

    # print(f'\nSłownik: {classes_dictionary}\n'
    #       f'Najczęściej występujący klasyfikator: {most_common_classifier}\n'
    #       f'Liczba wystąpień klasyfikatora: {largest_number_of_classifiers}\n'
    #       f'Ilość powtórzeń: {repeat}\n')

    # Jeżeli ilość powtrzórzeń będzię większa niż 1 ...
    if repeat > 1:
        # to zwróć 'REMIS':
        print(f'REMIS: Jest więcej niż jedno maksimum. Brak decyzji.')
        return 'REMIS'
    else:
        # Inaczej zwróć klasyfikator:
        print(f'Predykcja: {most_common_classifier}')
        return most_common_classifier


def first_knn_variant(dataset, sample, k_value, metrick_function):
    """
        Wariant I - jeżeli jest klasa, która ma największą częstość wtedy zostanie sklasyfikowana.
    """

    sorted_distances_list = calculate_distances(dataset, sample, metrick_function)
    neighbors_list = k_nearest_neighbors(sorted_distances_list, k_value)
    classes_list = classes_function(neighbors_list)

    return classes_list


def second_knn_variant(dataset, sample, k_value, metrick_function):
    """
        Wariant II - jeżeli isniteje klasa, która ma najmniejszą sumę wtedy można sklasyfikować.
    """

    # Lista unikalnych klasyfikatorów:
    unique_classifiers_list = np.unique(dataset[:, -1])

    # Słownik na klasyfikatory i ich odległości:
    unique_classifiers_dictionary = dict()

    # Wprowadzanie klasyfiaktorów do słownika jako klucze:
    for classifier in unique_classifiers_list:
        unique_classifiers_dictionary[classifier] = list()

    # Posortowana lista wierszy datasetu i ich odległości od próbki:
    sorted_distances_list = calculate_distances(dataset, sample, metrick_function)

    # Przypisanie odległości do odpowiadajacych im klasyfikaotrów:
    for element in sorted_distances_list:
        if element[-1][-1] in unique_classifiers_list:
            unique_classifiers_dictionary[element[-1][-1]].append(element[0])

    # Słownik na zsumowane odległości klasyfikatorów:
    distances_sums_dictionary = dict()

    # Wprowadzanie klasyfiaktorów do słownika jako klucze:
    for classifier in unique_classifiers_list:
        distances_sums_dictionary[classifier] = list()

    # Przypisanie k dystansów do odpowiedniego klasyfikatora:
    for key in unique_classifiers_dictionary.keys():
        for distance in range(k_value):
            distances_sums_dictionary[key].append(unique_classifiers_dictionary[key][distance])

    # Zsumowanie wartości odległości dla klasyfikaotrów:
    for key, value in distances_sums_dictionary.items():
        distances_sums_dictionary[key] = sum(value)

    # Ilość powtórzeń minimalnych odległości:
    repeat = 0

    # Klasyfikator z najmniejszą odległością:
    classifier_shortest_distance = min(distances_sums_dictionary, key=distances_sums_dictionary.get)

    # Wartość najmniejszej odległości:
    shortest_distance = min(distances_sums_dictionary.values())

    # Zwiększenie licznika powtórzeń najmniejszej wartości odległości:
    for element in distances_sums_dictionary.values():
        if element == shortest_distance:
            repeat += 1

    # print(f'\nMinimum: {shortest_distance}\n'
    #       f'Ilość powtórzeń: {repeat}')

    # Jeżeli ilość powtrzórzeń będzię większa niż 1 ...
    if repeat > 1:
        # to zwróć 'REMIS':
        print(f'REMIS: Jest więcej niż jedno minimum. Brak decyzji.')
        return 'REMIS'
    else:
        # Inaczej zwróć klasyfikator:
        print(f'Predykcja: {classifier_shortest_distance}')
        return classifier_shortest_distance


class WindowApp(Frame):
    def __init__(self, container):
        super(WindowApp, self).__init__(container)

        self.creating_window()       # Stworzenie elemwntów w oknnie aplikacji
        self.grid()                 # Umieszczenie elementów w oknie aplikacji

    def select_file(self):
        """
            Funkcja służy do wczytania pliku gotowego do działania KNNa (po normalizacji).
        """

        self.dataframe = None

        self.filepath = filedialog.askopenfilename(title='Wybierz plik')
        print(f'\nŚcieżka do pliku: {self.filepath}\n')

        if os.path.isfile(self.filepath):

            separator = extracting_separator(self.filepath)

            try:
                self.dataframe = pd.read_csv(self.filepath, header=None, encoding='utf-8', sep=separator)
            except ValueError:
                print('\nTo nie plik CSV.\n')

            self.dataset_name = datafame_validation(self.dataframe)

            self.display_dataframe_textbox.delete('1.0', END)
            self.display_dataframe_textbox.insert(END, self.dataframe)

    def select_new_row(self):
        """
            Funkcja służy do wczytania pliku zawierający nowy wiersz.
        """

        self.new_row = None

        self.filepath = filedialog.askopenfilename(title='Wybierz plik')
        print(f'\nŚcieżka do pliku: {self.filepath}\n')

        if os.path.isfile(self.filepath):

            separator = extracting_separator(self.filepath)

            try:
                self.new_row = pd.read_csv(self.filepath, header=None, encoding='utf-8', sep=separator)
            except ValueError:
                print('\nTo nie plik CSV.\n')

            self.new_row_filename = new_row_validation(self.new_row)
            if self.new_row_filename != self.dataset_name:
                print(f'\nPodany wiersz nie pasuje do datasetu: \'{self.dataset_name}\'\n')
                del self.new_row_filename
            else:
                print(f'\nPodany wiersz pasuje do datasetu: \'{self.dataset_name}\'\n')
                self.display_new_row_textbox.delete('1.0', END)
                self.display_new_row_textbox.insert(END, self.new_row.iloc[:, :-1])

    def get_selected_metric(self):
        """
            Funkcja ma zwrócić jaka metoda obliczania odległości została wybrana.
        """

        metric = None

        if self.select_metric.get() == 'euclidean':
            metric = 'euclidean'
        elif self.select_metric.get() == 'manhattan':
            metric = 'manhattan'
        elif self.select_metric.get() == 'hamming':
            metric = 'hamming'
        elif self.select_metric.get() == 'chebyshev':
            metric = 'chebyshev'

        print(f'\nWybrana metryka: {metric}\n')
        return metric

    def get_selected_sample(self):
        """
            Funkcja ma wyciągać podaną wartość próbki.
        """
        sample = self.select_sample_entry.get()

        try:
            sample = int(sample)

            print(f'\nWybrana próbka: {sample}\n')
            return sample
        except ValueError:
            print(f'\nPodana wartość {sample} nie jest liczbą.\n')
        except IndexError:
            print(f'\nNie istnieje próbka o numerze {sample}.\n')

    def get_selected_k_value(self):
        """
            Funkcja ma wyciągać podaną wartość k.
        """

        k = self.select_k_value_entry.get()

        try:
            k = int(k)

            if k <= 1:
                message = f'\nk nie może być mniejsze równe 1.\n'
                print(f'{message}')
                # messages_textbox.delete('1.0', END)
                # messages_textbox.insert(END, message)

            # elif k % 2 == 0 or k < 3:
            #     message = f'k powinno być liczbą całkowita nieparzystą większa równą 3.\n'
            #     print(f'{message}')
            #     # messages_textbox.delete('1.0', END)
            #     # messages_textbox.insert(END, message)

            else:
                message = f'\nPodana wartość k = {k}.\n'
                print(f'{message}')
                # messages_textbox.delete('1.0', END)
                # messages_textbox.insert(END, message)

                # print(global_k)
                return k

        except ValueError:
            message = f'\nPodana wartość k = {k} nie jest liczbą całkowitą (int).\n'
            print(f'{message}')
            # messages_textbox.delete('1.0', END)
            # messages_textbox.insert(END, message)

    def get_selected_variant(self):
        """
            Funkcja odpowiada za wybór wariantu KNN.
        """

        if self.select_variant.get() == 'KNN-1':
            self.first_knn_variant()
        elif self.select_variant.get() == 'KNN-2':
            self.second_knn_variant()

    def first_knn_variant(self):
        """
            Fukcja do predyckji klasyfikatora.
        """

        sample_value = self.get_selected_sample()
        k_value = self.get_selected_k_value()
        metric_function = self.get_selected_metric()

        sample = list(self.dataframe.iloc[sample_value])
        dataset = self.dataframe.to_numpy()

        result = first_knn_variant(dataset, sample, k_value, metric_function)

        self.classification_textbox.delete(0.0, END)
        self.classification_textbox.insert(END, result)

    def second_knn_variant(self):

        """
            Fukcja do predyckji klasyfikatora.
        """

        sample_value = self.get_selected_sample()
        k_value = self.get_selected_k_value()
        metric_function = self.get_selected_metric()

        sample = list(self.dataframe.iloc[sample_value])
        dataset = self.dataframe.to_numpy()

        result = second_knn_variant(dataset, sample, k_value, metric_function)

        self.classification_textbox.delete(0.0, END)
        self.classification_textbox.insert(END, result)

    def covering_accuracy_function(self):
        """
            Jeżeli na 100 obiektów (wierszy) udało się sklasyfikować 85 z nich to pokrycie:

            85/100 = 85% pokrycia

            W 85 obiektach (wierszach), które udało się sklasyfikować sprawdzamy czy klasyfikacja
            dała wynik identyczny jak klasa decyzyjna zapisana w tym obiekcie, np.
            80 obiektów z 85 zostało sklasyfikowane poprawnie, wtedy:

            80/85 = ok.94% skuteczności

            Predykcja       |   Pokrycie    |   Skuteczność
            ☁ == ☁                +1                +1
            ☁ == ☀                +1                +0
            ☁ == 'REMIS'          +0                +0
        """

        variant_function = self.select_variant.get()
        k_value = self.get_selected_k_value()
        metric_function = self.get_selected_metric()
        dataset = self.dataframe.to_numpy()

        if variant_function == 'KNN-1':
            good_prediction_counter = 0     # Trafione predykcje
            bad_prediction_counter = 0      # Nietrafione predykcje
            draw_prediction_counter = 0     # Brak predykcji

            dataset_size = len(dataset)     # Wielkość całego datasetu
            cover = 0                       # Pokrycie
            accuracy = 0                    # Skuteczność

            for row in dataset:
                prediction = first_knn_variant(dataset, row, k_value, metric_function)

                if prediction == row[-1]:
                    good_prediction_counter += 1
                    cover += 1
                    accuracy += 1
                elif prediction != row[-1] and prediction != 'REMIS':
                    bad_prediction_counter += 1
                    cover += 1
                    accuracy += 0
                elif prediction == 'REMIS':
                    draw_prediction_counter += 1
                    cover += 0
                    accuracy += 0

            cov = (cover / dataset_size) * 100
            acc = (accuracy / dataset_size) * 100

            print(f'\nIlość trafionych predykcji: {good_prediction_counter}\n'
                  f'Ilość nietrafionych predykcji: {bad_prediction_counter}\n'
                  f'Ilość remisów: {draw_prediction_counter}\n'
                  f'Pokrycie: {cover} / {dataset_size} = {cov}\n'
                  f'Skuteczność: {accuracy} / {dataset_size} = {acc}\n')

            self.covering_textbox.delete(0.0, END)
            self.covering_textbox.insert(END, f'{round(cov, 2)}%')
            self.accuracy_textbox.delete(0.0, END)
            self.accuracy_textbox.insert(END, f'{round(acc, 2)}%')

        elif variant_function == 'KNN-2':
            good_prediction_counter = 0     # Trafione predykcje
            bad_prediction_counter = 0      # Nietrafione predykcje
            draw_prediction_counter = 0     # Brak predykcji

            dataset_size = len(dataset)     # Wielkość całego datasetu
            cover = 0                       # Pokrycie
            accuracy = 0                    # Skuteczność

            for row in dataset:
                prediction = second_knn_variant(dataset, row, k_value, metric_function)

                if prediction == row[-1]:
                    good_prediction_counter += 1
                    cover += 1
                    accuracy += 1
                elif prediction != row[-1] and prediction != 'REMIS':
                    bad_prediction_counter += 1
                    cover += 1
                    accuracy += 0
                elif prediction == 'REMIS':
                    draw_prediction_counter += 1
                    cover += 0
                    accuracy += 0

            cov = (cover / dataset_size) * 100
            acc = (accuracy / dataset_size) * 100

            print(f'\nIlość trafionych predykcji: {good_prediction_counter}\n'
                  f'Ilość nietrafionych predykcji: {bad_prediction_counter}\n'
                  f'Ilość remisów: {draw_prediction_counter}\n'
                  f'Pokrycie: {cover} / {dataset_size} = {cov}\n'
                  f'Skuteczność: {accuracy} / {dataset_size} = {acc}\n')

            self.covering_textbox.delete(0.0, END)
            self.covering_textbox.insert(END, f'{round(cov, 2)}%')
            self.accuracy_textbox.delete(0.0, END)
            self.accuracy_textbox.insert(END, f'{round(acc, 2)}%')

    def decision_for_new_row(self):
        # if self.dataframe != None:
        variant_function = self.select_variant.get()        # KNN1 lub KNN2
        k_value = self.get_selected_k_value()               # Wartość k
        metric_function = self.get_selected_metric()        # Metryka
        dataset = self.dataframe.to_numpy()                 # Znormalizowane dane

        sample = list(self.new_row)

        if variant_function == 'KNN-1':
            result = first_knn_variant(dataset, sample, k_value, metric_function)
            print(f'Decyzja dla nowego wiersza: {result}')

            self.decision_for_new_row_textbox.delete(0.0, END)
            self.decision_for_new_row_textbox.insert(END, result)
        elif variant_function == 'KNN-2':
            result = second_knn_variant(dataset, sample, k_value, metric_function)
            print(f'Decyzja dla nowego wiersza: {result}')

            self.decision_for_new_row_textbox.delete(0.0, END)
            self.decision_for_new_row_textbox.insert(END, result)
        # else:
        #     print(f'Przed podjęciem decyzji należy zebrać dane.')

    # Stworzenie okna:
    def creating_window(self):
        # WYBÓR PLIKU: =================================================================================================

        file_select_labelframe = LabelFrame(self, text='1. Wybierz plik:')
        file_select_labelframe.grid(padx=2, pady=2, sticky='NW')

        Button(file_select_labelframe, text='...', command=self.select_file).grid(padx=2, pady=2)

        # WYŚWIETLENIE WCZYTANYCH DANYCH: ==============================================================================

        display_dataframe_labelframe = LabelFrame(self, text='Dane:')
        display_dataframe_labelframe.grid(padx=2, pady=2, sticky='NW')

        self.display_dataframe_textbox = Text(display_dataframe_labelframe, width=128, height=16, wrap=CHAR)
        self.display_dataframe_textbox.grid(padx=2, pady=2)

        # PANEL OPCJI: =================================================================================================

        options_setting_labelframe = LabelFrame(self, text='Opcje:')
        options_setting_labelframe.grid(padx=2, pady=2, sticky='NW')

        # WYBÓR WARIANTU KNN: ------------------------------------------------------------------------------------------

        self.variants_to_select_list = ['Pierwszy wariant KNN', 'Drugi wariant KNN']

        self.select_variant = StringVar()
        self.select_variant.set('KNN-1')        # Ustawienie wartości domyślnej: KNN-1

        variant_selection_labelframe = LabelFrame(options_setting_labelframe, text='2. Wybierz wariant:')
        variant_selection_labelframe.grid(column=0, row=0, padx=2, pady=2)

        Radiobutton(variant_selection_labelframe, text='Pierwszy wariant KNN', variable=self.select_variant, value='KNN-1').grid(padx=2, pady=2)
        Radiobutton(variant_selection_labelframe, text='Drugi wariant KNN', variable=self.select_variant, value='KNN-2').grid(padx=2, pady=2)

        # WYBÓR METRYKI: -----------------------------------------------------------------------------------------------

        self.select_metric = StringVar()
        self.select_metric.set('euclidean')     # Ustawienie wartości domyślnej: euclidean

        metric_selection_labelframe = LabelFrame(options_setting_labelframe, text='3. Wybierz metodę:')
        metric_selection_labelframe.grid(column=1, row=0, padx=2, pady=2)

        Radiobutton(metric_selection_labelframe, text='euclidean', variable=self.select_metric, value='euclidean').grid(padx=2, pady=2)
        Radiobutton(metric_selection_labelframe, text='manhattan', variable=self.select_metric, value='manhattan').grid(padx=2, pady=2)
        Radiobutton(metric_selection_labelframe, text='hamming', variable=self.select_metric, value='hamming').grid(padx=2, pady=2)
        Radiobutton(metric_selection_labelframe, text='chebyshev', variable=self.select_metric, value='chebyshev').grid(padx=2, pady=2)

        # PODAJ WARTOŚĆ K: ---------------------------------------------------------------------------------------------

        select_k_value_labelframe = LabelFrame(options_setting_labelframe, text='4. Podaj wartość k:')
        select_k_value_labelframe.grid(column=2, row=0, padx=2, pady=2)

        self.select_k_value_entry = Entry(select_k_value_labelframe)
        self.select_k_value_entry.grid(padx=2, pady=2)

        # PODANIE NUMERU PRÓBKI: ---------------------------------------------------------------------------------------

        select_sample_labelframe = LabelFrame(options_setting_labelframe, text='5. Podaj numer próbki:')
        select_sample_labelframe.grid(column=3, row=0, padx=2, pady=2)

        self.select_sample_entry = Entry(select_sample_labelframe)
        self.select_sample_entry.grid(padx=2, pady=2)

        # KLASYFIKACJA: ================================================================================================

        classification_labelframe = LabelFrame(self, text='Klasyfikacja:')
        classification_labelframe.grid(padx=2, pady=2, sticky='NW')

        self.classification_textbox = Text(classification_labelframe, height=1, wrap=CHAR)
        self.classification_textbox.grid(column=0, row=0, padx=2, pady=2)

        Button(classification_labelframe, text='Zatwierdź', command=self.get_selected_variant).grid(column=1, row=0, padx=2, pady=2)

        # STATYSTYKA & POKRYCIE & SKUTECNOŚĆ: ==========================================================================

        statistics_labelframe = LabelFrame(self, text='6. Pokrycie & Skuteczność:')
        statistics_labelframe.grid(padx=2, pady=2, sticky='NW')

        Button(statistics_labelframe, text='Wykonaj statystykę', command=self.covering_accuracy_function).grid(padx=2, pady=2)

        self.covering_textbox = Text(statistics_labelframe, height=1, wrap=CHAR)      # Pokrycie
        self.covering_textbox.grid(padx=2, pady=2)
        self.accuracy_textbox = Text(statistics_labelframe, height=1, wrap=CHAR)      # Trafność/Skuteczność
        self.accuracy_textbox.grid(padx=2, pady=2)

        # DECYZJA DLA NOWEGO WIERSZA: ==================================================================================

        decision_for_new_row_labelframe = LabelFrame(self, text='7. Decyzja dla nowego wiersza')
        decision_for_new_row_labelframe.grid(padx=2, pady=2, sticky='NW')

        Button(decision_for_new_row_labelframe, text='...', command=self.select_new_row).grid(column=1, row=0, padx=2, pady=2)
        Button(decision_for_new_row_labelframe, text='Zatwierdź', command=self.decision_for_new_row).grid(column=1, row=1, padx=2, pady=2)

        self.display_new_row_textbox = Text(decision_for_new_row_labelframe, width=128, height=2, wrap=CHAR)
        self.display_new_row_textbox.grid(column=0, row=0, padx=2, pady=2)

        self.decision_for_new_row_textbox = Text(decision_for_new_row_labelframe, width=128, height=1, wrap=CHAR)
        self.decision_for_new_row_textbox.grid(column=0, row=1, padx=2, pady=2)


window = Tk()
window.title('Zadanie 2')
window_app = WindowApp(window)
window.mainloop()
