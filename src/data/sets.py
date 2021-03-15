def split_sets_random(df, target_col, test_ratio=0.2, to_numpy=False):
    """Split sets randomly

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    
    from sklearn.model_selection import train_test_split
    
    features, target = pop_target(df=df, target_col=target_col, to_numpy=to_numpy)
    
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8)
    
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8)

    return X_train, y_train, X_val, y_val, X_test, y_test

def pop_target(df, target_col, to_numpy=False):
    """Extract target variable from dataframe and convert to nympy arrays if required

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable
    to_numpy : bool
        Flag stating to convert to numpy array or not

    Returns
    -------
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing all features
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    
    if to_numpy:
        df_copy = df_copy.to_numpy()
        target = target.to_numpy()
    
    return df_copy, target
