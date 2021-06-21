import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#creating a function
def data_prep():
    # read the veteran dataset
    df = pd.read_csv('clinical.csv')

    #remove whitespace from column names
    df.columns=df.columns.str.strip()
    
    #create a dataframe with important features
    df = df[['Patient age quantile', 'SARS-Cov-2 exam result', 'Hematocrit','Hemoglobin', 'Platelets', 'Mean platelet volume',
         'Red blood Cells', 'Lymphocytes', 'Mean corpuscular hemoglobin concentration (MCHC)', 'Leukocytes', 'Basophils', 
         'Mean corpuscular hemoglobin (MCH)', 'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes', 
         'Red blood cell distribution width (RDW)', 'Proteina C reativa mg/dL', 'Neutrophils','Influenza B, rapid test', 
         'Influenza A, rapid test']].copy()

    # change Influenza variables into binary 0/1 variable
    influenza_map = {'negative':0, 'positive': 1}
    df['Influenza B, rapid test'] = df['Influenza B, rapid test'].map(influenza_map)
    df['Influenza A, rapid test'] = df['Influenza A, rapid test'].map(influenza_map)
    
    #need to map SARS-Cov-2 exam result
    COVID_map = {'negative':0, 'positive': 1}
    df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].map(COVID_map)

    # impute missing values with its mean
    df['Hematocrit'].fillna(df['Hematocrit'].mean(), inplace=True)
    df['Hemoglobin'].fillna(df['Hemoglobin'].mean(), inplace=True)
    df['Platelets'].fillna(df['Platelets'].mean(), inplace=True)
    df['Mean platelet volume'].fillna(df['Mean platelet volume'].mean(), inplace=True)
    df['Red blood Cells'].fillna(df['Red blood Cells'].mean(), inplace=True)
    df['Lymphocytes'].fillna(df['Lymphocytes'].mean(), inplace=True)
    df['Mean corpuscular hemoglobin concentration (MCHC)'].fillna(df['Mean corpuscular hemoglobin concentration (MCHC)'].mean(), inplace=True)
    df['Leukocytes'].fillna(df['Leukocytes'].mean(), inplace=True)
    df['Basophils'].fillna(df['Basophils'].mean(), inplace=True)
    df['Mean corpuscular hemoglobin (MCH)'].fillna(df['Mean corpuscular hemoglobin (MCH)'].mean(), inplace=True)
    df['Eosinophils'].fillna(df['Eosinophils'].mean(), inplace=True)
    df['Mean corpuscular volume (MCV)'].fillna(df['Mean corpuscular volume (MCV)'].mean(), inplace=True)
    df['Monocytes'].fillna(df['Monocytes'].mean(), inplace=True)
    df['Red blood cell distribution width (RDW)'].fillna(df['Red blood cell distribution width (RDW)'].mean(), inplace=True)
    df['Proteina C reativa mg/dL'].fillna(df['Proteina C reativa mg/dL'].mean(), inplace=True)
    df['Neutrophils'].fillna(df['Neutrophils'].mean(), inplace=True)

    # remove rows with missing values
    df.dropna(inplace = True)

    # one-hot encoding
    df = pd.get_dummies(df)
    
    # target/input split
    y = df['SARS-Cov-2 exam result']
    X = df.drop(['SARS-Cov-2 exam result'], axis=1)

    # setting random state
    rs = 10

    X_mat = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    return df,X,y,X_train, X_test, y_train, y_test