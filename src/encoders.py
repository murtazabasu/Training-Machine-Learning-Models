from sklearn import preprocessing

"""
This file contains the encoder to be used
to train the model.   
"""

encoders = {
    "one_hot_encoding": preprocessing.LabelEncoder(),
    "label_encoding": preprocessing.OneHotEncoder(),
}