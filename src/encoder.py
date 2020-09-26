from sklearn import preprocessing

"""
This file contains the encoder to be used
to train the model.   
"""

encoders = {
    "label_encoding": preprocessing.LabelEncoder(),
    "one_hot_encoding": preprocessing.OneHotEncoder(),
}