import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from models.rf import random_forest
from utils import clean_mutations


## Loading data once
mutations_raw = pd.read_csv('../data/raw/CosmicMutantExportCensus.tsv', delimiter="\t")


def train_model(XX, yy, models):
    # Split dataset for trainging and test
    X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2)
    # Train model
    model = random_forest(X_train, y_train, n_est=10)
    # Test model
    y_score = model.predict(X_test)
    # Get model stats
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # Append model stats to return list
    models.append([fpr, tpr, roc_auc, model, XX])
    # Get mode model stats and print
    tn, fp, fn, tp = confusion_matrix(y_test, y_score).ravel()
    print("AUC :", roc_auc)
    print("False Negative: ", fn)
    print("False Negative Rate: ", fn/(fn+tp))
    print("False Positive: ", fp)
    print("False Positive Rate: ", fp/(tn+fp))
    print("True Negative: ", tn)
    print("True Negative Rate: ", tn/(tn+fp))
    print("True Positive: ", tp)
    print("True Positive Rate: ", tp/(fn+tp))


def generate_models(cancer, headers=[], num=3):
    # Filter by cancer type
    mutations = mutations_raw.loc[mutations_raw['Primary site'] == cancer]
    # Remove unnecessary columns
    clean_mutations(mutations)
    # Fill Resistance Mutation nulls
    mutations['Resistance Mutation'].replace(to_replace=['-'], value=0, inplace=True)
    mutations['Resistance Mutation'].replace(to_replace=['Yes'], value=1, inplace=True)
    # Create new dataset with a count of appearence of each Mutation ID
    tmp_df = mutations.groupby(['Mutation ID']).size().reset_index(name="count").sort_values(by="count", ascending=False)
    # Create a new dataset with only mutations that has more than one appearence
    tmp2_df = tmp_df.loc[tmp_df['count'] > 1]
    # Merge count information with the full dataset
    mutations = pd.merge(mutations, tmp2_df, on="Mutation ID", how="left")
    # Exclude from full dataset tummors with mutations that are unique
    mutations = mutations.loc[mutations['count'] > 1]
    # Flat Mutations into rows
    dummies = pd.get_dummies(mutations['Mutation ID'], dummy_na=True)
    result = pd.concat([mutations, dummies], axis=1)
    # Remove Mutation ID column
    result.drop('Mutation ID', axis=1, inplace=True)
    # Fill Age nulls with overall mean
    result['Age'] = result['Age'].fillna(round(result['Age'].mean(), 0))
    # Fill FATHMM prediction nulls with Unsure as it's a 0,5 case
    result['FATHMM prediction'] = result['FATHMM prediction'].fillna("Unsure")
    # Remove unnecessari columns for training
    del result['ID_sample']
    del result['count']
    # Map FATHMM prediction to numbers asuming PATHOGENIC and Unsure are quite the same
    result['FATHMM prediction'].replace(to_replace=['PATHOGENIC'], value=2, inplace=True)
    result['FATHMM prediction'].replace(to_replace=['Unsure'], value=1, inplace=True)
    result['FATHMM prediction'].replace(to_replace=['NEUTRAL'], value=0, inplace=True)

    # Split dataset from results for training
    X = result.copy()
    del X['Resistance Mutation']
    y = result['Resistance Mutation']
    # Keep models track for return
    models = []
    if len(headers) > 0:
        X = X[headers]
    # Train num models
    for n in range(num):
        train_model(X, y, models)
    # Return models
    return models
