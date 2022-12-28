import math
from random import shuffle

from tkinter import *
from tkinter import filedialog
import numpy as np
import pandas as pd
import csv
import json
import os

"""
    Zadanie 3:
    1.1) Wygenerować odpowiednią ilosc losowych danych
       z zadanego zakresu na podstawie struktury, np.:
       3 - 2 - 2
       4 - 3 - 2 - 3 - 10
    2.1) Dane zapisac w taki sposób, aby nie stracić
       informacji o strukturze.
    3.1) Odczytac dane z pliku.

    1.2) Funkcjonalności z zadania 1b
        a) generowanie wag dla wskazanej struktury
        b) zapisywanie wag do pliku (z podaniem nazwy, aby można było zapisać więcej niż jeden plik)
        c) odczytanie wag z plików wcześniej zapisanych
    2.2) Możliwość zadawania pytań do sieci
        a) bezpośrednio w interfejsie użytkownika
        b) po przez plik
    3.2) Podawanie próbek uczących próbka ucząca, to zestaw wejść i wyjść liczebnie zależny od struktury).
         Proponuje podawać je w pliku, gdyż próbek uczących w zależności od struktury sieci może być bardzo dużo.
         a) próbki uczące do uczenia się muszą być podawane w sposób losowy
    4.2) Ustawianie parametrów uczenia się
        a) B - beta (można na sztywno)
        b) mi - współczynnik uczenia się (można na sztywno)
    5.2) Warunki stopu uczenia się przez propagację wsteczną (oba muszą być zaimplementowane, użytkownik wybiera, którego chce użyć)
        a) Zatrzymanie uczenia, po przekroczeniu założonej liczby iteracji (ile razy powtarzamy propagację wsteczną)
        b) Zatrzymanie uczenia, po osiągnięciu błędu sieci mniejszego niż założony
           (błąd powinien być liczony jako średnia z ostatnich wyników, najlepiej liczba wyników jako parametr do ustalenia)
        c) Możliwość zapisu wag po procesie uczenia

    1.1. ✅
    2.1. ✅
    3.1. ✅

    1.2.
        a. ✅
        b. ✅
        c. ✅
    2.2.
        a. ✅
        b. ✅
    3.2. ✅
        a. ✅
    4.2.
        a. ✅
        b. ✅
    5.2.
        a. ✅
        b. ✅
        c. ✅
"""


# FUNKCJE: #############################################################################################################

def print_structure_dictionary(dictionary):
    """
        Funkcja służy do wyświetlania w konsoli wygenerowanych wartości dla podanej struktury.
    """
    print(f'\nStrukruta ➤ {dictionary["structure-list"]}')

    for layer in dictionary['structure-dict']:
        print(f'L#{layer} ↴')
        for neuron in dictionary['structure-dict'][layer]:
            print(f'\tN#{neuron} ↴')
            for weight in dictionary['structure-dict'][layer][neuron]:
                print(f'\t\tW#{weight}: {dictionary["structure-dict"][layer][neuron][weight]}')


def create_list_from_structure_dictionary(dictionary):
    """
    Stworzenie listy wartości wag z słownika.
    :param dictionary:
    :return: Lista wszyskich wag z struktury.
    """
    weights_list = list()

    for layer in dictionary:
        for neuron in dictionary[layer]:
            for weight in dictionary[layer][neuron]:
                weights_list.append(dictionary[layer][neuron][weight])

    return weights_list


def create_list_list_list(structure_list, structure_dictionary):
    """
    Stworzenie listy list list odpowiadającej słownikowi struktury.
    :param structure_list:
    :param structure_dictionary:
    :return:    L3 - słownik struktury przekonwertowany na listy
    """
    weights_list = create_list_from_structure_dictionary(structure_dictionary)
    list_list_list = list()  # L3[między warstwa][węzeł po prawej stronie][krawędź] = waga

    for between_layer in range(0, len(structure_list) - 1):
        list_list_list.append(list())
        for right_side_node in range(0, structure_list[between_layer + 1]):
            list_list_list[between_layer].append(list())
            for _ in range(0, structure_list[between_layer] + 1):
                list_list_list[between_layer][right_side_node].append(weights_list[0])
                weights_list.pop(0)

    return list_list_list


def extracting_separator(path):
    """
        Funkcja ma za zadanie określenie separatora danych w podanum pliku.

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


def training_samples_preparation(dataset, inputs):
    """
        Przykładowy zbiór danych (dataset):
        0 0 0
        0 1 1
        1 0 1
        1 1 0

        PARY ➤ 2 wejścia (inputs) & 1 wyjście (outputs):
        [[0, 0], [0]]
        [[0, 1], [1]]
        [[1, 0], [1]]
        [[1, 1], [0]]
    """
    dataframe_tolist = dataset.values.tolist()

    inputs_list = list()
    outputs_list = list()
    dataset_list = list()

    for index in range(len(dataframe_tolist)):
        dataset_list.append(list())
        inputs_list.append(list())
        outputs_list.append(list())

    for index in range(len(dataframe_tolist)):
        for jndex in range(len(dataframe_tolist[index])):
            if jndex < inputs:
                inputs_list[index].append(dataframe_tolist[index][jndex])

    for index in range(len(dataframe_tolist)):
        for jndex in range(len(dataframe_tolist[index])):
            if jndex >= inputs:
                outputs_list[index].append(dataframe_tolist[index][jndex])

    for index in range(len(dataframe_tolist)):
        dataset_list[index].append(inputs_list[index])
        dataset_list[index].append(outputs_list[index])

    return dataset_list


def save_as_JSON(path, data):
    with open(str(path), 'w') as file:
        json.dump(data, file)


# IMPLEMNTACJA SIECI: ##################################################################################################

class Neuron:
    def __init__(self, network_structure, inputs_number, BIAS, BETA):
        """
        :param network_structure (dict):  Struktura sieci, np.: [3 4 2 1] + Wygenerowane wagi dla struktury
        :param inputs_number (int):       Liczba węzłów z poprzedniej wasrtwy , np.: 3
        :param BIAS (int):                Wartość BIAS, np.: 1
        :param BETA (int):                Wartość BETA, np.: 1

        Neuron ↴
            WEIGHTS:    [0.1, 0.3, 0.4, 0.1]
            OUTPUT:     0.0
            DELTA:      0.0

            BIAS:       1
            BETA:       1
        """
        weight_list = create_list_from_structure_dictionary(network_structure)

        self.weights = list()
        for _ in range(inputs_number + 1):              # INPUTS + BIAS
            self.weights.append(weight_list[0])
            weight_list.pop(0)

        weight_list = create_list_from_structure_dictionary(network_structure)

        self.BIAS = BIAS
        self.BETA = BETA

        self.output = 0.0   # Obliczona wartość OUTPUT z propagacji w przód
        self.delta = 0.0    # Obliczona wartość błądu dla propagacji w tył

        # print(f'\nN ➤ W:\t{self.weights}')

    def activation(self, weights, inputs):
        """
        Obliczenie aktywacji neuronu.
        :param weights:     Wagi przypisane do neuronu
        :param inputs:      Wartości wejściowe neuronu
        :return:            Zwraca obliczoną sumę przemnożonych wag (WEIGHTS) i wejść (INPUTS) neuronu

        Przykład:
        WEIGHTS = [0.1, 0.3, 0.5]    INPUTS = [0 1]    BIAS = 1
        activation = (weights[0] * inputs[0]) + (weights[1] * inputs[1]) + (weights[2] * BIAS)
        activation = (0.1 * 0) + (0.3 * 1) + (0.5 * 1) = 0.8
        """
        activation = self.BIAS * weights[-1]  # Ostania waga * BIAS

        for index in range(len(weights) - 1):
            activation = activation + inputs[index] * weights[index]

        return activation

    def sigmoid(self, activation):
        """
        Funkcja aktwacji - SIGMOID
        :param activation:  Aktywacja neuronu - obliczona suma przemnożonych wag (WEIGHTS) i wejść (INPUTS) neuronu
        :return:            Zwraca wartość wyjściową (OUTPUT) neuronu

        Przykład:
        activation = 0.8    BETA = 1
        sigmoid = 1 / (1 + e^(-BETA * activation))
        sigmoid = 1 / (1 + e^(-1 * 0.8)) = 0.6899745
        """
        sigmoid = 1.0 / (1.0 + math.exp(-self.BETA * activation))
        # print(f'\nN ➤ SIGMOID:\t{sigmoid}')
        return sigmoid

    def sigmoid_derivative(self, output):
        """
        Obliczenie pochodnej
        :param output:  Wartość wyjściowa (OUTPUT) neuronu
        :return:        Zwrócenie pochodnej z wartości wyjściowej (OUTPUT) neuronu

        output = 0.6899745
        derivative = output * (1.0 - output)
        derivative = 0.6899745 * (1.0 - 0.6899745) = 0.21390968934975
        """
        derivative = output * (1.0 - output)
        # print(f'\nN ➤ POCHODNA: {derivative}')
        return derivative


class Layer:
    def __init__(self, network_structure, neurons_number, inputs_number, BIAS, BETA):
        """
        :param network_structure:   Struktura sieci, np.: [3 4 2 1]
        :param neurons_number:      Liczba węzłów w warstwie, np. 4
        :param inputs_number:       Liczba węzłów z poprzedniej wasrtwy, np. 3
        :param BIAS:                Wartość BIAS
        :param BETA:                Wartość BETA

        Layer = [N0, N1, itd.]
        """
        self.neurons_list = list()

        for _ in range(0, neurons_number):
            self.neurons_list.append(Neuron(network_structure, inputs_number, BIAS, BETA))


class Network:
    def __init__(self, network_structure, inputs=3, hidden=[3, 2], outputs=2, BIAS=1, BETA=1):
        """
        :param network_structure:   Struktura sieci, np.: [3 4 2 1]
        :param inputs:              Ilość węzłów odpowiadająca za wejścia, np. 3
        :param hidden:              Lista warstw ukrytych, np. [4 2]
        :param outputs:             Ilość węzłów odpowiadająca za wyjścia, np. 1
        :param BIAS:                Wartość BIAS
        :param BETA:                Wartość BETA

        LW:
        [
            L0:
            [
                N0: { W: [(...)], O: forward(), D: back() },
                N1: { W: [(...)], O: forward(), D: back() },
                N2: { W: [(...)], O: forward(), D: back() }
            ]

            L1:
            [
                N0: { W: [(...)], O: forward(), D: back() },
                N1: { W: [(...)], O: forward(), D: back() },
                N2: { W: [(...)], O: forward(), D: back() }
            ]

            L2:
            [
                N0: { W: [(...)], O: forward(), D: back() },
                N1: { W: [(...)], O: forward(), D: back() },
            ]
        ]
        """

        self.network_structure = network_structure
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.BIAS = BIAS
        self.BETA = BETA

        self.layers = [inputs] + hidden + [outputs]     # Struktura, np.: [3 4 2 1]
        self.layers_list = list()                       # Lista dla międzywarstw

        for index in range(1, len(self.layers)):
            self.layers_list.append(Layer(network_structure, self.layers[index], self.layers[index - 1], BIAS, BETA))

    def forward_propagate(self, row):
        """
        Obliczanie wartości wyjściowych (neuron.output) dla neuronów.
        :param row:     Wiersz wartości wejściowych (INPUTS)
        :return:        Zwraca wartości wyjściowe ostatniej wartswy (OUTPUT LAYER)
        """
        inputs = row
        # print(f'\nFP➤INPUTS:{inputs}\n')

        for index, layer in enumerate(self.layers_list):
            # print(f'FP➤L{index}:{layer}')
            new_inputs = list()     # Lista dla nowych wartości wejściowych (INPUTS)

            for jndex, neuron in enumerate(layer.neurons_list):
                # print(f'FP➤N{jndex}:{neuron}')
                # for kndex, weight in enumerate(neuron.weights):
                #     print(f'FP➤W{kndex}:{weight}')

                activation = neuron.activation(neuron.weights, inputs)      # Aktywacja neuronu
                # print(f'FP➤ACT:{activation}')
                neuron.output = neuron.sigmoid(activation)                  # Wartość wyjściowa neuronu
                # print(f'FP➤N➤OUT:{neuron.output}')

                # Wartości wyjściowe neuronów stają się wartościami wejściowymi neuronów z nastepnej warstwy:
                new_inputs.append(neuron.output)

            inputs = new_inputs
            # print(f'\nFP➤NEW INPUTS:{inputs}')

        return inputs

    def backword_propagation(self, targets):
        """
        Obliczanie sygnałów błędów (neuron.delta) dla neuronów.
        :param targets:      Wartości oczekiwane
        """
        for index in reversed(range(len(self.layers_list))):
            layer = self.layers_list[index]
            errors_list = list()

            # Dla pozostałych warstw:
            if index != len(self.layers_list) - 1:
                for jndex in range(len(layer.neurons_list)):
                    error = 0.0

                    for kndex, neuron in enumerate(self.layers_list[index + 1].neurons_list):
                        # Przemnażamy wagi neuronu z wastwy następnej przez wcześniej obliczoną deltę
                        error += (neuron.weights[jndex] * neuron.delta)
                    errors_list.append(error)
            # Dla warstwy ostatniej:
            else:
                for jndex in range(len(layer.neurons_list)):
                    neuron = layer.neurons_list[jndex]
                    error = neuron.output - targets[jndex]
                    errors_list.append(error)

            for jndex in range(len(layer.neurons_list)):
                neuron = layer.neurons_list[jndex]          # Przypisanie do zmiennej neuron neuronu z warstwy
                neuron.delta = errors_list[jndex] * neuron.sigmoid_derivative(neuron.output)

    # AKTUALIZACJA WAG: ------------------------------------------------------------------------------------------------

    def update_weights(self, row, learning_rate):
        """
        Aktualizacja wag sieci.
        :param row:             Wartości IMPUT z wastwy INPUT LAYER
        :param learning_rate:   Wartość współczynnika uczenia, np.: 0.1

        weight = weight - learning_rate * error * input
        * weight - dana waga
        * lr - współczynnik uczenia
        * error - błąd obliczony propagacją wsteczną dla neuronu
        * input - wartość, która spowodowała błąd
        """

        for index in range(len(self.layers_list)):
            inputs = row    # Wartości wejściowe z pierwszej warstwy

            if index != 0:
                inputs = [neuron.output for neuron in self.layers_list[index - 1].neurons_list]

            # print(f'L{index}: {inputs}')

            for neuron in self.layers_list[index].neurons_list:
                for jndex in range(len(inputs)):
                    # Poprawka dla wag:
                    neuron.weights[jndex] = neuron.weights[jndex] - learning_rate * neuron.delta * inputs[jndex]
                # Poprawka dla wag ukrytych:
                neuron.weights[-1] = neuron.weights[-1] - learning_rate * neuron.delta * self.BIAS

    def save_weights(self):
        """
        Przygotowanie zmieniownych wag do zapisu.
        :return:    `final_structure_dict` - ULTIMATE SUPER SIGMA SAIYAN STRUCTURE DICT
        """

        new_network_structure = self.network_structure
        save_weights = list()

        for layer in self.layers_list:
            # print(f'L: {layer}')
            for neuron in layer.neurons_list:
                # print(f'N: {neuron}')
                for weight in neuron.weights:
                    # print(f'W: {weight}')
                    save_weights.append(weight)

        # print(f'\nWAGI DO ZAPISANIA:\n{save_weights}')

        for layer in new_network_structure:
            # print(f'SW ➤ FP ➤ L{layer} ↴')
            for neuron in new_network_structure[layer]:
                # print(f'\tSW ➤ FP ➤ N{neuron} ↴')
                for weight in new_network_structure[layer][neuron]:
                    new_network_structure[layer][neuron][weight] = save_weights[0]
                    save_weights.pop(0)
                    # print(f'\t\tSW ➤ FP ➤ W{weight}:\t{new_network_structure[layer][neuron][weight]}')

        final_structure_dict = \
            {
                'structure-list': self.layers,
                'structure-dict': new_network_structure
            }

        return final_structure_dict

    # TRNING: ----------------------------------------------------------------------------------------------------------

    def mean_squared_error(self, targets, outputs):
        """
        Obliczenie średniej wartości błędu dla predykcji (outputs) i wartości oczekiwanych (targets)
        Przykład:
        Error = (output[0] - target[0]) ^ 2
        Error = (0.734682 - 1)^2 = 0.070393641124
        """
        error = list()
        for index in range(len(targets)):
            error.append((outputs[index] - targets[index]) ** 2)

        mse = sum(error)    # Suma błędów
        return mse

    def train_epoch(self, dataset, learning_rate, epochs):
        """
        Trening sieci przez określoną liczbę cykli.
        :param dataset:         Zbiór próbek uczących
        :param learning_rate:   Wartość współczynnika uczenia, np.: 0.1
        :param epochs:          Ilość cykli do przeprowadzenia, np.: 1000
        """
        inputs_index = 0
        targets_index = 1

        print(f'WSPÓŁCZYNNIK UCZENIA: {learning_rate}')

        for epoch in range(epochs):
            sum_error = 0.0
            shuffle(dataset)  # Pomieszanie próbek

            for row in dataset:
                inputs = row[inputs_index]
                targets = row[targets_index]

                # print(f'INPUTS: {inputs} ➤ TARGETS: {targets}')

                outputs = self.forward_propagate(inputs)            # Wartości wyjściowe sieci
                mse = self.mean_squared_error(targets, outputs)     # Średnia wartość błędu
                sum_error = sum_error + mse                         # Błąd sieci

                self.backword_propagation(targets)                  # Obliczenie błędów dla neuronów
                self.update_weights(inputs, learning_rate)          # Aktualizacja wag sieci

            print(f'CYKL: {epoch + 1}\t|\tBŁĄD: {sum_error}')

        # for index, layer in enumerate(self.layers_list):
        #     print(f'\nL#{index} ↴')
        #     for jndex, neuron in enumerate(layer.neurons_list):
        #         print(f'\tN#{jndex} ↴\n'
        #               f'\t\tW: {neuron.weights}\n'
        #               f'\t\tO: {neuron.output}\n'
        #               f'\t\tD: {neuron.delta}\n')

    def train_error_value(self, dataset, learning_rate, error_value):
        """
        Trening sieci aż do osiągnięcia wartości błędu mniejszej niż wskazana.
        :param dataset:         Zbiór próbek uczących
        :param learning_rate:   Wartość współczynnika uczenia, np.: 0.1
        :param error_value:     Wartość błędu do którego ma dążyć błąd
        """
        inputs_index = 0
        targets_index = 1

        print(f'WSPÓŁCZYNNIK UCZENIA: {learning_rate}')

        tmp_error = error_value + 1
        epoch = 0

        while error_value < tmp_error:
            sum_error = 0.0
            shuffle(dataset)  # Pomieszanie próbek

            for row in dataset:
                inputs = row[inputs_index]
                targets = row[targets_index]

                # print(f'INPUTS: {inputs} ➤ TARGETS: {targets}')

                outputs = self.forward_propagate(inputs)
                mse = self.mean_squared_error(targets, outputs)
                sum_error = sum_error + mse

                self.backword_propagation(targets)
                self.update_weights(inputs, learning_rate)

            tmp_error = sum_error
            print(f'CYKL: {epoch + 1}\t|\tBŁĄD: {sum_error}')
            epoch += 1

            # if epoch == 100000:
            #     break

        # for index, layer in enumerate(self.layers_list):
        #     print(f'\nL#{index} ↴')
        #     for jndex, neuron in enumerate(layer.neurons_list):
        #         print(f'\tN#{jndex} ↴\n'
        #               f'\t\tW: {neuron.weights}\n'
        #               f'\t\tO: {neuron.output}\n'
        #               f'\t\tD: {neuron.delta}\n')

    # PREDYKCJA: -------------------------------------------------------------------------------------------------------

    def activation(self, weights, inputs):
        activation = self.BIAS * weights[-1]  # Ostania waga * BIAS

        for index in range(len(weights) - 1):
            activation = activation + inputs[index] * weights[index]

        return activation

    def sigmoid(self, activation):

        sigmoid = 1.0 / (1.0 + math.exp(-self.BETA * activation))
        return sigmoid

    def propagate(self, structure_list, structure_dictionary, row):
        weights_list = create_list_list_list(structure_list, structure_dictionary)
        inputs = row

        for index, layer in enumerate(weights_list):
            new_inputs = list()

            for jndex, neuron in enumerate(layer):
                activation = self.activation(neuron, inputs)
                output = self.sigmoid(activation)
                new_inputs.append(output)

            inputs = new_inputs

        return inputs

    def predict(self, structure_list, structure_dictionary, row):
        outputs = self.propagate(structure_list, structure_dictionary, row)
        print(f'\nPredykcja dla {row} ➤ {outputs}')

        return outputs


# APLIKACJA: ###########################################################################################################


class WindowApp(Frame):
    def __init__(self, container):
        super(WindowApp, self).__init__(container)

        self.creating_window()  # Stworzenie elemwntów w oknnie aplikacji
        self.grid()  # Umieszczenie elementów w oknie aplikacji

        self.structure_list = list()
        self.structure_dict = dict()
        self.filepath = None
        self.data_file_path = None
        self.load_json_file = dict()

        self.question_filepath = None
        self.question_list = None

        self.dataframe = None
        self.training_samples = None

        self.network = None
        self.epochs = 10
        self.learning_rate = 0.1
        self.BIAS = 1
        self.BETA = 1

    def get_structure_list(self):
        """
            Funkcja wyciąga wartości wprowadzone przez użytkownika przy generowaniu struktury.
        """

        get_structure = self.structure_entry.get()

        structure_list = get_structure.split(' ')
        structure_list = list(map(int, structure_list))

        return structure_list

    def create_structure(self):
        """
            Funkcja odpowiada za wygenerowanie wartości dla struktury.

            * structure_list - lista int wprowadzona przez użytkownika
            * structure_dict - słownik z wygenerowanymi wartościami dla struktury
            * final_structure_dict - połączenie `structure_list` i `structure_dict` zapisane do pliku *.JSON
        """
        structure_list = self.get_structure_list()
        range_from = int(self.weights_range_from_entry.get())
        range_to = int(self.weights_range_to_entry.get())

        structure_dict = dict()

        for between_layer in range(0, len(structure_list) - 1):
            structure_dict[between_layer] = dict()
            for right_side_node in range(0, structure_list[between_layer + 1]):
                structure_dict[between_layer][right_side_node] = dict()
                for edge in range(0, structure_list[between_layer] + 1):
                    structure_dict[between_layer][right_side_node][edge] = np.random.uniform(range_from, range_to)

        final_structure_dict = \
            {
                'structure-list': structure_list,
                'structure-dict': structure_dict
            }

        # Wyświetlenie stworzonej struktury w konsoli:
        print_structure_dictionary(final_structure_dict)

        filetypes_list = [('JSON', '*.json')]

        filename = filedialog.asksaveasfilename(filetypes=filetypes_list, defaultextension=json)

        save_as_JSON(filename, final_structure_dict)

    # WCZYTANIE STRUKTURY:

    def load_structure(self):
        file_types_list = [('JSON', '*.json')]

        self.filepath = filedialog.askopenfilename(filetypes=file_types_list)
        print(f'\nŚcieżka pliku: {self.filepath}\n')

        if os.path.isfile(self.filepath):
            with open(self.filepath, 'r') as file:
                self.load_json_file = json.load(file)

            print_structure_dictionary(self.load_json_file)

            self.structure_list = self.load_json_file['structure-list']

            messege = f'\n{self.structure_list}\n'
            self.messages_textbox.insert(END, messege)

            try:
                self.structure_targest_dict = self.load_json_file['unique_targets_dict']
                messege = f'\n{self.structure_targest_dict}\n'
                self.messages_textbox.insert(END, messege)
            except:
                pass

            try:
                self.structure_beta = self.load_json_file['BETA']
                messege = f'\n{self.structure_beta}\n'
                self.messages_textbox.insert(END, messege)
            except:
                pass

            self.structure_dict = self.load_json_file['structure-dict']

            for index, layer in self.structure_dict.items():
                for jndex, neuron in layer.items():
                    for kndex, weight in neuron.items():
                        messege = f'L#{index}\tN#{jndex}\tW#{kndex}: {weight}\n'
                        self.messages_textbox.insert(END, messege)

    # WCZYTANIE PRÓBEK UCZĄCYCH:

    def to_float(self, samples):
        for index in range(len(samples)):
            for jndex in range(len(samples[index])):
                samples[index][jndex] = float(samples[index][jndex])

        # for row in dataset:
        #     row[column] = float(row[column].strip())

        return samples

    def targets_values(self, targets):
        targets_values = list()

        for index in range(len(targets)):
            for jndex in range(len(targets[index])):
                targets_values.append(targets[index][jndex])

        unique_values = set(targets_values)
        new_targets = dict()  # Słownik zawierający stare i nowe wartości TARGET

        for index, value in enumerate(unique_values):
            new_targets[value] = index

        for index in range(len(targets)):
            for jndex in range(len(targets[index])):
                targets[index][jndex] = new_targets[targets[index][jndex]]

        return new_targets

    def samples_minmax(self, samples):
        minmax = list()

        for column in zip(*samples):
            minmax.append([min(column), max(column)])

        return minmax

    def normalize_samples(self, samples, minmax):
        # formula = (value - minimum) / (maximum - minimum)         0-1
        # formula = formula * (range_to - range_from) + range_from  inaczej

        for row in samples:
            for index in range(len(row)):
                row[index] = (row[index] - minmax[index][0]) / (minmax[index][1] - minmax[index][0])

        return samples

    def load_training_samples(self):
        data_file_types = [('DATA', '*.data')]

        self.data_file_path = filedialog.askopenfilename(filetypes=data_file_types)
        print(f'\nŚcieżka pliku: {self.data_file_path}\n')

        if os.path.isfile(self.data_file_path):
            separator = extracting_separator(self.data_file_path)

            try:
                self.dataframe = pd.read_csv(self.data_file_path, header=None, encoding='utf-8', sep=separator)
            except ValueError:
                print(f'Nie udało się wczytać pliku. Spróbuj ponownie z innym plikiem.')

        # print(f'\nDATAFRAME:\n{self.dataframe}')
        self.training_samples = training_samples_preparation(self.dataframe, self.structure_list[0])

        samples = list()  # Wartości próbek
        targets = list()  # Wartości oczekiwane/klasy itd.

        for index in range(len(self.training_samples)):
            samples.append(self.training_samples[index][0])
            targets.append(self.training_samples[index][1])

        # NORMALIZACJA: ------------------------------------------------------------------------------------------------

        samples = self.to_float(samples)

        self.unique_targets_dict = self.targets_values(targets)

        minmax = self.samples_minmax(samples)

        normalize_samples = self.normalize_samples(samples, minmax)

        self.new_training_samples = list()

        for index in range(len(self.training_samples)):
            self.new_training_samples.append(list())

        for index in range(len(self.training_samples)):
            self.new_training_samples[index].append(normalize_samples[index])
            self.new_training_samples[index].append(targets[index])

        # KOMUNIKAT: ---------------------------------------------------------------------------------------------------

        message = f'\nPróbki uczące:\n'
        # self.messages_textbox.delete('1.0', END)
        self.messages_textbox.insert(END, message)

        for index in range(len(self.new_training_samples)):
            message = f'{self.new_training_samples[index]}\n'
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

    # FUNKCJE WYCIĄGANIA WPROWADZONYCH WARTOŚCI: =======================================================================

    def get_epoch_value(self):
        e = self.epochs_error_value_entry.get()

        try:
            e = int(e)

            if e < 10:
                message = f'\nIlość cykli przemiału danych nie może być mniejsza niż 10'
                print(f'{message}')
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

            else:
                message = f'\nIlość epok: {e}'
                print(f'{message}')
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

                return e

        except ValueError:
            message = f'\nPodana wartość dla epok = {e} nie jest liczbą całkowitą (int).\n'
            print(f'{message}')
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

    def get_error_value(self):
        e = self.epochs_error_value_entry.get()

        try:
            e = float(e)

            message = f'\nBłąd sieci do osiągnięcia: {e}'
            print(f'{message}')
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

            return e

        except ValueError:
            message = f'\nPodana wartość dla błędu = {e} nie jest liczbą.'
            print(f'{message}')
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

    def get_learning_rate_value(self):
        lr = self.learning_rate_entry.get()

        try:
            lr = float(lr)

            if lr >= 1:
                message = f'\nWspółczynnik uczenia µ powinien być mniejszy niż 1\n'
                print(f'{message}')
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

            else:
                message = f'\nµ: {lr}\n'
                print(f'{message}')
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

                return lr

        except ValueError:
            message = f'\nPodana wartość dla współczynnika uczenia µ = {lr} nie jest liczbą.\n'
            print(f'{message}')
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

    def get_beta_value(self):
        b = self.beta_value_entry.get()

        try:
            b = int(b)

            if b < 1:
                message = f'\nBETA nie może być mniejsza niż 1.\n'
                print(f'{message}')
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

            else:
                message = f'\nβ: {b}'
                print(f'{message}')
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

                return b

        except ValueError:
            message = f'\nWartość musi być liczbą.\n'
            print(f'{message}')
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

    def get_question_value(self):
        q = self.question_entry.get()

        try:
            q = q.split(' ')
            q = list(float(element) for element in q)

            if len(q) > self.structure_list[0]:

                message = f'\nPrzekroczenie ilości wyjść.\n'
                # self.messages_textbox.delete('1.0', END)
                self.messages_textbox.insert(END, message)

            else:
                return q

        except ValueError:
            message = f'\nSpróbuj inaczej\n'
            print(f'{message}')
            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)

    def get_question_file(self):
        question_dataframe = None

        question_filetypes = [('DATA', '*data')]

        self.question_filepath = filedialog.askopenfilename(filetypes=question_filetypes)

        if os.path.isfile(self.question_filepath):
            separator = extracting_separator(self.question_filepath)

            try:
                question_dataframe = pd.read_csv(self.question_filepath, header=None, encoding='utf-8', sep=separator)

                if len(question_dataframe.columns) == self.structure_list[0]:
                    self.question_list = question_dataframe.values.tolist()
                else:
                    raise IndexError

            except ValueError:
                print(f'Nie udało się wczytać pliku. Spróbuj ponownie z innym plikiem.')
            except IndexError:
                print(f'Niewłaściwe pytanie. Zadaj inne.')

        print(f'\nPYTANIE/A Z PLIKU (DATAFRAME):\n{question_dataframe}')
        print(f'\nPYTANIE/A Z PLIKU (LIST): {self.question_list}')

        # return self.question_list

    def get_selected_train_method(self):
        if self.select_trening_method.get() == 'epochs':
            self.epochs_value_variant()
        elif self.select_trening_method.get() == 'error_value':
            self.error_value_variant()

    # METODY PRZEPROWADZENIA CYKLI TRANINGU SIECI: =====================================================================

    def epochs_value_variant(self):
        epochs = self.get_epoch_value()
        learning_rate = self.get_learning_rate_value()
        beta = self.get_beta_value()

        sl = self.structure_list  # Lista zawierająca strukturę
        sd = self.structure_dict  # Słownik zaiwerający wagi dla struktury sieci

        training_samples = self.new_training_samples

        network1 = Network(network_structure=sd, inputs=sl[0], hidden=sl[1:-1], outputs=sl[-1], BIAS=1, BETA=beta)
        network1.train_epoch(dataset=training_samples, learning_rate=learning_rate, epochs=epochs)

        save_weights = network1.save_weights()
        save_weights['unique_targets_dict'] = self.unique_targets_dict
        save_weights['BETA'] = beta

        filetypes_list = [('JSON', '*.json')]

        filename = filedialog.asksaveasfilename(filetypes=filetypes_list, defaultextension=json)

        save_as_JSON(filename, save_weights)

    def error_value_variant(self):
        error = self.get_error_value()
        learning_rate = self.get_learning_rate_value()
        beta = self.get_beta_value()

        sl = self.structure_list  # Lista zawierająca strukturę
        sd = self.structure_dict  # Słownik zaiwerający wagi dla struktury sieci

        training_samples = self.training_samples

        network2 = Network(network_structure=sd, inputs=sl[0], hidden=sl[1:-1], outputs=sl[-1], BIAS=1, BETA=beta)
        network2.train_error_value(dataset=training_samples, learning_rate=learning_rate, error_value=error)

        save_weights = network2.save_weights()
        save_weights['unique_targets_dict'] = self.unique_targets_dict
        save_weights['BETA'] = beta

        filetypes_list = [('JSON', '*.json')]

        filename = filedialog.asksaveasfilename(filetypes=filetypes_list, defaultextension=json)

        save_as_JSON(filename, save_weights)

    # PREDYKCJA: =======================================================================================================

    def corect_prediction(self, st, prediction):

        targets_index = list()
        targets_value = list()
        for key, value in st.items():
            targets_index.append(int(value))
            targets_value.append(int(key))

        range_list = np.arange(0, 1, (1 / len(prediction))).tolist()
        range_list.append(1)

        dictionary = dict()
        for index in range(len(targets_value)):
            dictionary[targets_value[index]] = list(range_list[index:index + 2])

        for index in range(len(prediction)):
            for target, range_list in dictionary.items():
                if range_list[0] <= prediction[index] <= range_list[1]:
                    prediction[index] = target

        return prediction

    def predict_answer_entry(self):
        question = self.get_question_value()
        # print(f'PA ➤ QUESTION:\t{question}')

        sl = self.structure_list
        sd = self.structure_dict
        st = self.structure_targest_dict
        sb = self.structure_beta

        network3 = Network(network_structure=sd, inputs=sl[0], hidden=sl[1:-1], outputs=sl[-1], BIAS=1, BETA=sb)
        prediction = network3.predict(sl, sd, question)

        answer = self.corect_prediction(st, prediction)

        message = f'\nP:{question} ➤ O:{answer}\n'
        # self.messages_textbox.delete('1.0', END)
        self.messages_textbox.insert(END, message)

    def predict_answer_file(self):
        question = self.question_list
        # print(f'PA ➤ QUESTION:\t{question}')

        sl = self.structure_list
        sd = self.structure_dict
        st = self.structure_targest_dict
        sb = self.structure_beta

        network3 = Network(network_structure=sd, inputs=sl[0], hidden=sl[1:-1], outputs=sl[-1], BIAS=1, BETA=sb)

        for index in range(len(question)):
            prediction = network3.predict(sl, sd, question[index])
            answer = self.corect_prediction(st, prediction)

            message = f'\nP:{question[index]} ➤ O:{answer}\n'

            # self.messages_textbox.delete('1.0', END)
            self.messages_textbox.insert(END, message)
        self.messages_textbox.insert(END, '\n')

    # TKINTER: =========================================================================================================

    def creating_window(self):

        messages_labelframe = LabelFrame(self, text='Komunikaty')
        messages_labelframe.grid(rowspan=4, column=1, padx=2, pady=2)

        self.messages_textbox = Text(messages_labelframe, height=50, width=150, wrap=CHAR)
        self.messages_textbox.grid(padx=2, pady=2, sticky=NS)

        # GENERATOR STRUKTURY: =========================================================================================

        generate_structure_labelframe = LabelFrame(self, text='GENERATOR STRUKTURY')
        generate_structure_labelframe.grid(row=0, column=0, padx=2, pady=2)

        # Wpisanie struktury: ------------------------------------------------------------------------------------------

        structure_labelframe = LabelFrame(generate_structure_labelframe, text='1. Wprowadź strukrurę:')
        structure_labelframe.grid(padx=2, pady=2)

        self.structure_entry = Entry(structure_labelframe)
        self.structure_entry.grid(padx=2, pady=2)

        # Wpisanie wag od do: ------------------------------------------------------------------------------------------

        weights_range_labelframe = LabelFrame(generate_structure_labelframe, text='2. Wprowadź zakres wag:')
        weights_range_labelframe.grid(padx=2, pady=2)

        Label(weights_range_labelframe, text='Od:').grid(row=0, column=0, padx=2, pady=2)
        self.weights_range_from_entry = Entry(weights_range_labelframe)
        self.weights_range_from_entry.grid(row=0, column=1, padx=2, pady=2)

        Label(weights_range_labelframe, text='Do:').grid(row=1, column=0, padx=2, pady=2)
        self.weights_range_to_entry = Entry(weights_range_labelframe)
        self.weights_range_to_entry.grid(row=1, column=1, padx=2, pady=2)

        Button(generate_structure_labelframe, text='3. Zatwierdź', command=self.create_structure).grid(padx=2, pady=2)

        # TRENING SIECI: ===============================================================================================

        train_network_labelframe = LabelFrame(self, text='TRENOWANIE SIECI')
        train_network_labelframe.grid(row=1, column=0, padx=2, pady=2)

        # Wczytanie struktury z pliku JSON: ----------------------------------------------------------------------------

        load_structure_labelframe = LabelFrame(train_network_labelframe, text='1. Wczytaj strukturę:')
        load_structure_labelframe.grid(padx=2, pady=2)

        Button(load_structure_labelframe, text='...', command=self.load_structure).grid(padx=2, pady=2)

        # Wczytanie próbek z pliku DATA: -------------------------------------------------------------------------------

        load_dataset_labelframe = LabelFrame(train_network_labelframe, text='2. Wczytaj próbki:')
        load_dataset_labelframe.grid(padx=2, pady=2)

        Button(load_dataset_labelframe, text='...', command=self.load_training_samples).grid(padx=2, pady=2)

        # Wprowadzenie parametów: --------------------------------------------------------------------------------------

        # Wybór metody przebiegu treningu:

        self.select_trening_method = StringVar()
        self.select_trening_method.set('epochs')

        trening_method_labelframe = LabelFrame(train_network_labelframe, text='3. Wybierz metodę treningu')
        trening_method_labelframe.grid(padx=2, pady=2)

        Radiobutton(trening_method_labelframe, text='Epoki', variable=self.select_trening_method, value='epochs').grid(
            padx=2, pady=2)
        Radiobutton(trening_method_labelframe, text='Wartość błędu', variable=self.select_trening_method,
                    value='error_value').grid(padx=2, pady=2)

        Label(trening_method_labelframe, text='Wprowadź wartość epok/błędu:').grid(padx=2, pady=2)
        self.epochs_error_value_entry = Entry(trening_method_labelframe)
        self.epochs_error_value_entry.grid(padx=2, pady=2)

        # Wprowadzenie współczynnika uczenia:

        Label(train_network_labelframe, text='4. Wprowadź µ:').grid(padx=2, pady=2)
        self.learning_rate_entry = Entry(train_network_labelframe)
        self.learning_rate_entry.grid(padx=2, pady=2)

        # Wprowadzenie BETY:

        Label(train_network_labelframe, text='5. Wprowadź β:').grid(padx=2, pady=2)
        self.beta_value_entry = Entry(train_network_labelframe)
        self.beta_value_entry.grid(padx=2, pady=2)

        # Zatwierdzenie i zapis do pliku:

        Button(train_network_labelframe, text='6. Zatwierdź', command=self.get_selected_train_method).grid(padx=2,
                                                                                                           pady=2)

        # PREDYKCJA: ===================================================================================================

        predict_labelframe = LabelFrame(self, text='PREDYKCJA')
        predict_labelframe.grid(row=3, column=0, padx=2, pady=2)

        # Wczytanie wyuczonej struktury sieci: -------------------------------------------------------------------------

        load_trained_structure_labelframe = LabelFrame(predict_labelframe, text='1. Wczytaj wyuczoną strukturę sieci:')
        load_trained_structure_labelframe.grid(padx=2, pady=2)

        Button(load_trained_structure_labelframe, text='...', command=self.load_structure).grid(padx=2, pady=2)

        # Wprowadzenie danych: -----------------------------------------------------------------------------------------

        ask_question_labelframe = LabelFrame(predict_labelframe, text='2. Wprowadź pytanie lub wczytaj je z pliku:')
        ask_question_labelframe.grid(padx=2, pady=2)

        # Wprowadzenie ręczne:

        ask_question_entry_labalframe = LabelFrame(ask_question_labelframe, text='Wprowadź pytanie:')
        ask_question_entry_labalframe.grid(row=0, column=0, padx=2, pady=2)

        self.question_entry = Entry(ask_question_entry_labalframe)
        self.question_entry.grid(padx=2, pady=2)

        Button(ask_question_entry_labalframe, text='Zatwierdź', command=self.predict_answer_entry).grid(padx=2, pady=2)

        # Wczytanie z pliku:

        ask_question_file_labelframe = LabelFrame(ask_question_labelframe, text='Wczytaj pytanie')
        ask_question_file_labelframe.grid(row=0, column=1, padx=2, pady=2)

        Button(ask_question_file_labelframe, text='...', command=self.get_question_file).grid(padx=2, pady=2)

        Button(ask_question_file_labelframe, text='Zatwierdź', command=self.predict_answer_file).grid(padx=2, pady=2)


window = Tk()
window.title('Zadanie 3')
window_app = WindowApp(window)
window.mainloop()
