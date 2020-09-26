import joblib
import argparse
import pandas as pd
import model_dispatcher
import config
import encoders
from sklearn import metrics


def run(df, fold, encoder, model):

    # treat all columns as features except id, target and kfold columns

    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    #fill all NaN values with none strings

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
        #Initialize the encoder for each column of the features
        enc = encoders.encoders[encoder]
        
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
    
    # initialize model
    model = model_dispatcher.models[model]

    # fit model on training data
    model.fit(x_train, df_train.target.values)

    # predict on validation data

    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc 
    print(f"Fold = {fold}, AUC = {auc}") 

    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )  

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder", type=str, )
    parser.add_argument("--model", type=str)
    parser.add_argument("--splits", type=int, help="input the number \
                        of splits for creating folds")

    # read the argument from the command line
    args = parser.parse_args()

    run(fold=args.fold, model=args.model)
    df = pd.read_csv(config.NEW_DATA)
    for fold_ in range(df.kfold.unique()):
        run(df, fold_, encoder=args.encoder, model=args.model)
                              








