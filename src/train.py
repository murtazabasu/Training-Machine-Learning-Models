import os
import joblib
import argparse
import pandas as pd

from arguments import get_args
from folds import create_folds
import model_dispatcher
import config
import encoder
from sklearn import metrics


def run(df, fold, args, encoder_, model_):

    # treat all columns as features except id, target and kfold columns

    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    #fill all NaN values with none strings
    if args.encoder == "label_encoding":
        for col in features:
            #Initialize the encoder for each column of the features
            enc = encoder.encoders[encoder_]
            
            #fit encoder on all data
            enc.fit(df[col])

            #transform all the data
            df.loc[:, col] = enc.transform(df[col])

        # get training data using folds
        df_train = df[df.kfold!=fold].reset_index(drop=True)

        # get validation data using folds
        df_valid = df[df.kfold==fold].reset_index(drop=True)

        # get training data
        x_train = df_train[features].values
        
        # get validation data
        x_valid = df_valid[features].values
    
    else:
        # get training data using folds
        df_train = df[df.kfold!=fold].reset_index(drop=True)
        # get validation data using folds
        df_valid = df[df.kfold==fold].reset_index(drop=True)

        enc = encoder.encoders[encoder_]

        concat_data = pd.concat(
                [df_train[features], df_valid[features]],
                axis=0
        )
        enc.fit(concat_data[features])

        # transform training and testing data
        x_train = enc.transform(df_train[features])
        x_valid = enc.transform(df_valid[features])
    
    # initialize model
    model = model_dispatcher.models[model_]

    # fit model on training data
    model.fit(x_train, df_train.target.values)

    # predict on validation data

    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc 
    if fold==0:
        print(f"\nModel-->{args.model}")
        print(f"Encoder-->{args.encoder}")
    print(f"Fold = {fold}, AUC = {auc}") 

    joblib.dump(
        model, 
        os.path.join(config.MODEL_OUTPUT, f"fold_{fold}_{args.model}_{args.encoder}.bin")
    )  

if __name__=="__main__":
    args = get_args()
    
    if args.create_folds:
        create_folds(args.n_splits)

    df = pd.read_csv(config.NEW_DATA)
    for fold_ in range(len(df.kfold.unique())):
        run(df, fold_, args, encoder_=args.encoder, model_=args.model)
                              








