FILE = 'CVD_Cleaned.csv'  # File location (CVD)

SAMPLE = 10000
TRAIN = 0.6
VALIDATION = 0.2

FEATURES = 18  # Number of features

EPOCHS = 50  # Epochs for training neural network
BATCH_SIZE = 2  # Number of items in a batch in data loader
HIDDEN_LAYER_SIZE = 20  # Number of nodes in hidden layers
LEARNING_RATE = 1e-3


# Numerical values for categorical attributes
translations_checkup = {
    'Within the past year': 1,
    'Within the past 2 years': 2,
    'Within the past 5 years': 3,
    '5 or more years ago': 4,
    'Never': 5
}
translations_age = {
    '18-24': 22,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-64': 62,
    '65-69': 67,
    '70-74': 72,
    '75-79': 77,
    '80+': 82
}
translations_general_health = {
    'Poor': 0,
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Excellent': 4
}
translations_diabetes = {'Yes': 'Yes', 'No': 'No',
                         'No, pre-diabetes or borderline diabetes': 'No',
                         'Yes, but female told only during pregnancy': 'Yes'}
