australianLink = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'
creditApprovalLink = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
breastCancerWisconsinLink = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast' \
                            '-cancer-wisconsin.data '

australianPath = 'data/australian.dat'
creditApprovalPath = 'data/crx.data'
breastCancerWisconsinPath = 'data/breast-cancer-wisconsin.data'

australianFileName = 'australian.dat'
creditApprovalFileName = 'crx.data'
breastCancerWisconsinFileName = 'breast-cancer-wisconsin.data'

datasets = \
    {
        'Australian':
            {
                'link': australianLink,
                'path': australianPath,
                'separator': ' ',
                'missing_values': [],
                'columns_types':
                    {
                        0: int,
                        1: float,
                        2: float,
                        3: int,
                        4: int,
                        5: int,
                        6: int,
                        7: float,
                        8: int,
                        9: int,
                        10: int,
                        11: int,
                        12: int,
                        13: int,
                        14: int,
                    },
                'drop_columns': [10, 13],
                'non-numeric_columns': [],
                'normalization_columns': [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                'decision_class': [-1]
            },

        'Credit-Approval':
            {
                'link': creditApprovalLink,
                'path': creditApprovalPath,
                'separator': ',',
                'missing_values': ['?'],
                'columns_types':
                    {
                        0: str,
                        1: float,
                        2: float,
                        3: str,
                        4: str,
                        5: str,
                        6: str,
                        7: float,
                        8: str,
                        9: str,
                        10: int,
                        11: str,
                        12: str,
                        13: int,
                        14: int,
                        15: str
                    },
                'drop_columns': [10, 13, 14],
                'non-numeric_columns': [0, 1, 3, 4, 5, 6, 8, 9, 11, 12],
                'normalization_columns': [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                'decision_class': [-1]
            },

        'Breast-Cancer-Wisconsin':
            {
                'link': breastCancerWisconsinLink,
                'path': breastCancerWisconsinPath,
                'separator': ',',
                'missing_values': ['?'],
                'columns_types':
                    {
                        0: int,
                        1: int,
                        2: int,
                        3: int,
                        4: int,
                        5: int,
                        6: int,
                        7: int,
                        8: int,
                        9: int,
                        10: int
                    },
                'drop_columns': [0],
                'non-numeric_columns': [],
                'normalization_columns': [1, 2, 3, 4, 5, 6, 8, 9],
                'decision_class': [-1]
            }
    }

# Schematy: ============================================================================================================

schemaAustralian = \
    {
        0: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
            'coerce': int},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'min': 1, 'max': 3, 'allowed': [number for number in range(1, 4)],
            'coerce': int},
        4: {'type': 'number', 'nullable': True, 'min': 1, 'max': 14, 'allowed': [number for number in range(1, 15)],
            'coerce': int},
        5: {'type': 'number', 'nullable': True, 'min': 1, 'max': 9, 'allowed': [number for number in range(1, 10)],
            'coerce': int},
        6: {'type': 'number', 'nullable': True, 'coerce': float},
        7: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
            'coerce': int},
        8: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
            'coerce': int},
        9: {'type': 'number', 'nullable': True, 'coerce': int},
        10: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
             'coerce': int},
        11: {'type': 'number', 'nullable': True, 'min': 1, 'max': 3, 'allowed': [number for number in range(1, 4)],
             'coerce': int},
        12: {'type': 'number', 'nullable': True, 'coerce': int},
        13: {'type': 'number', 'nullable': True, 'coerce': int},
        14: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
             'coerce': int},
    }

schemaCreditApproval = \
    {
        0: {'type': 'string', 'nullable': True, 'allowed': ['b', 'a']},
        1: {'type': 'float', 'nullable': True, 'coerce': float},
        2: {'type': 'float', 'nullable': True, 'coerce': float},
        3: {'type': 'string', 'nullable': True, 'allowed': ['u', 'y', 'l', 't']},
        4: {'type': 'string', 'nullable': True, 'allowed': ['g', 'p', 'gg']},
        5: {'type': 'string', 'nullable': True,
            'allowed': ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff']},
        6: {'type': 'string', 'nullable': True, 'allowed': ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o']},
        7: {'type': 'float', 'nullable': True, 'coerce': float},
        8: {'type': 'string', 'nullable': True, 'allowed': ['t', 'f']},
        9: {'type': 'string', 'nullable': True, 'allowed': ['t', 'f']},
        10: {'type': 'integer', 'nullable': True, 'coerce': int},
        11: {'type': 'string', 'nullable': True, 'allowed': ['t', 'f']},
        12: {'type': 'string', 'nullable': True, 'allowed': ['g', 'p', 's']},
        13: {'type': 'integer', 'nullable': True, 'coerce': int},
        14: {'type': 'integer', 'nullable': True, 'coerce': int},
        15: {'type': 'string', 'nullable': True, 'allowed': ['+', '-']}
    }

schemaBreastCancerWisconsin = \
    {
        0: {'type': 'integer', 'nullable': True, 'coerce': int},
        1: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        2: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        3: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        4: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        5: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        6: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        7: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        8: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        9: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        10: {'type': 'integer', 'nullable': True, 'min': 2, 'max': 4, 'allowed': [number for number in range(2, 5, 2)],
             'coerce': int},
    }

datasets_schemas = \
    {
        'Australian': schemaAustralian,
        'Credit-Approval': schemaCreditApproval,
        'Breast-Cancer-Wisconsin': schemaBreastCancerWisconsin
    }

# DATASETY PO NORMALIZACJI: ============================================================================================

normalized_datasets = \
    {
        'Australian':
            {
                'link': australianLink,
                'path': australianPath,
                'separator': ' ',
                'missing_values': [],
                'columns_types':
                    {
                        0: float,
                        1: float,
                        2: float,
                        3: float,
                        4: float,
                        5: float,
                        6: float,
                        7: float,
                        8: float,
                        9: float,
                        10: float,
                        11: float,
                        12: int,
                    },
                'drop_columns': [10, 13],
                'non-numeric_columns': [],
                'normalization_columns': [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                'decision_class': [-1]
            },

        'Credit-Approval':
            {
                'link': creditApprovalLink,
                'path': creditApprovalPath,
                'separator': ',',
                'missing_values': ['?'],
                'columns_types':
                    {
                        0: float,
                        1: float,
                        2: float,
                        3: float,
                        4: float,
                        5: float,
                        6: float,
                        7: float,
                        8: float,
                        9: float,
                        10: float,
                        11: float,
                        12: str
                    },
                'drop_columns': [10, 13, 14],
                'non-numeric_columns': [0, 1, 3, 4, 5, 6, 8, 9, 11, 12],
                'normalization_columns': [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                'decision_class': [-1]
            },

        'Breast-Cancer-Wisconsin':
            {
                'link': breastCancerWisconsinLink,
                'path': breastCancerWisconsinPath,
                'separator': ',',
                'missing_values': ['?'],
                'columns_types':
                    {
                        0: float,
                        1: float,
                        2: float,
                        3: float,
                        4: float,
                        5: float,
                        6: int,
                        7: float,
                        8: float,
                        9: int
                    },
                'drop_columns': [0],
                'non-numeric_columns': [],
                'normalization_columns': [1, 2, 3, 4, 5, 6, 8, 9],
                'decision_class': [-1]
            }
    }

# ZNORMALIZOWANE SCHEMATY: =============================================================================================

normalized_schemaAustralian = \
    {
        0: {'type': 'number', 'nullable': True, 'coerce': float},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'coerce': float},
        4: {'type': 'number', 'nullable': True, 'coerce': float},
        5: {'type': 'number', 'nullable': True, 'coerce': float},
        6: {'type': 'number', 'nullable': True, 'coerce': float},
        7: {'type': 'number', 'nullable': True, 'coerce': float},
        8: {'type': 'number', 'nullable': True, 'coerce': float},
        9: {'type': 'number', 'nullable': True, 'coerce': float},
        10: {'type': 'number', 'nullable': True, 'coerce': float},
        11: {'type': 'number', 'nullable': True, 'coerce': float},
        12: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
             'coerce': int}
    }

normalized_schemaCreditApproval = \
    {
        0: {'type': 'number', 'nullable': True, 'coerce': float},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'coerce': float},
        4: {'type': 'number', 'nullable': True, 'coerce': float},
        5: {'type': 'number', 'nullable': True, 'coerce': float},
        6: {'type': 'number', 'nullable': True, 'coerce': float},
        7: {'type': 'float', 'nullable': True, 'coerce': float},
        8: {'type': 'number', 'nullable': True, 'coerce': float},
        9: {'type': 'number', 'nullable': True, 'coerce': float},
        10: {'type': 'number', 'nullable': True, 'coerce': float},
        11: {'type': 'number', 'nullable': True, 'coerce': float},
        12: {'type': 'string', 'nullable': True, 'allowed': ['+', '-']}
    }

normalized_schemaBreastCancerWisconsin = \
    {
        0: {'type': 'number', 'nullable': True, 'coerce': float},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'coerce': float},
        4: {'type': 'number', 'nullable': True, 'coerce': float},
        5: {'type': 'number', 'nullable': True, 'coerce': float},
        6: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        7: {'type': 'number', 'nullable': True, 'coerce': float},
        8: {'type': 'number', 'nullable': True, 'coerce': float},
        9: {'type': 'integer', 'nullable': True, 'min': 2, 'max': 4, 'allowed': [number for number in range(2, 5, 2)],
            'coerce': int},
    }

normalized_datasets_schemas = \
    {
        'Australian': normalized_schemaAustralian,
        'Credit-Approval': normalized_schemaCreditApproval,
        'Breast-Cancer-Wisconsin': normalized_schemaBreastCancerWisconsin
    }

# DATASETY DLA NOWEGO WIERSZA: =========================================================================================

new_row_datasets = \
    {
        'Australian':
            {
                'link': australianLink,
                'path': australianPath,
                'separator': ' ',
                'missing_values': [],
                'columns_types':
                    {
                        0: float,
                        1: float,
                        2: float,
                        3: float,
                        4: float,
                        5: float,
                        6: float,
                        7: float,
                        8: float,
                        9: float,
                        10: float,
                        11: float,
                        12: int
                    },
                'drop_columns': [10, 13],
                'non-numeric_columns': [],
                'normalization_columns': [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                'decision_class': [-1]
            },

        'Credit-Approval':
            {
                'link': creditApprovalLink,
                'path': creditApprovalPath,
                'separator': ',',
                'missing_values': ['?'],
                'columns_types':
                    {
                        0: float,
                        1: float,
                        2: float,
                        3: float,
                        4: float,
                        5: float,
                        6: float,
                        7: float,
                        8: float,
                        9: float,
                        10: float,
                        11: float,
                        12: str
                    },
                'drop_columns': [10, 13, 14],
                'non-numeric_columns': [0, 1, 3, 4, 5, 6, 8, 9, 11, 12],
                'normalization_columns': [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12],
                'decision_class': [-1]
            },

        'Breast-Cancer-Wisconsin':
            {
                'link': breastCancerWisconsinLink,
                'path': breastCancerWisconsinPath,
                'separator': ',',
                'missing_values': ['?'],
                'columns_types':
                    {
                        0: float,
                        1: float,
                        2: float,
                        3: float,
                        4: float,
                        5: float,
                        6: int,
                        7: float,
                        8: float,
                        9: int
                    },
                'drop_columns': [0],
                'non-numeric_columns': [],
                'normalization_columns': [1, 2, 3, 4, 5, 6, 8, 9],
                'decision_class': [-1]
            }
    }

# SCHEMATY DLA NOWEGO WIERSZA: =========================================================================================

new_row_schemaAustralian = \
    {
        0: {'type': 'number', 'nullable': True, 'coerce': float},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'coerce': float},
        4: {'type': 'number', 'nullable': True, 'coerce': float},
        5: {'type': 'number', 'nullable': True, 'coerce': float},
        6: {'type': 'number', 'nullable': True, 'coerce': float},
        7: {'type': 'number', 'nullable': True, 'coerce': float},
        8: {'type': 'number', 'nullable': True, 'coerce': float},
        9: {'type': 'number', 'nullable': True, 'coerce': float},
        10: {'type': 'number', 'nullable': True, 'coerce': float},
        11: {'type': 'number', 'nullable': True, 'coerce': float},
        12: {'type': 'number', 'nullable': True, 'min': 0, 'max': 1, 'allowed': [number for number in range(0, 2)],
             'coerce': int}
    }

new_row_schemaCreditApproval = \
    {
        0: {'type': 'number', 'nullable': True, 'coerce': float},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'coerce': float},
        4: {'type': 'number', 'nullable': True, 'coerce': float},
        5: {'type': 'number', 'nullable': True, 'coerce': float},
        6: {'type': 'number', 'nullable': True, 'coerce': float},
        7: {'type': 'float', 'nullable': True, 'coerce': float},
        8: {'type': 'number', 'nullable': True, 'coerce': float},
        9: {'type': 'number', 'nullable': True, 'coerce': float},
        10: {'type': 'number', 'nullable': True, 'coerce': float},
        11: {'type': 'number', 'nullable': True, 'coerce': float},
        12: {'type': 'string', 'nullable': True, 'allowed': ['+', '-']}
    }

new_row_schemaBreastCancerWisconsin = \
    {
        0: {'type': 'number', 'nullable': True, 'coerce': float},
        1: {'type': 'number', 'nullable': True, 'coerce': float},
        2: {'type': 'number', 'nullable': True, 'coerce': float},
        3: {'type': 'number', 'nullable': True, 'coerce': float},
        4: {'type': 'number', 'nullable': True, 'coerce': float},
        5: {'type': 'number', 'nullable': True, 'coerce': float},
        6: {'type': 'integer', 'nullable': True, 'min': 1, 'max': 10, 'allowed': [number for number in range(1, 11)],
            'coerce': int},
        7: {'type': 'number', 'nullable': True, 'coerce': float},
        8: {'type': 'number', 'nullable': True, 'coerce': float},
        9: {'type': 'integer', 'nullable': True, 'min': 2, 'max': 4, 'allowed': [number for number in range(2, 5, 2)],
            'coerce': int},
    }

new_row_datasets_schemas = \
    {
        'Australian': new_row_schemaAustralian,
        'Credit-Approval': new_row_schemaCreditApproval,
        'Breast-Cancer-Wisconsin': new_row_schemaBreastCancerWisconsin
    }
