from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, fitted=False):
    '''
    Evaluates a model on train and test sets. 
    Returns a dictionary with metrics (Accuracy, ROC-AUC, F1-Score) 
    '''
    if not fitted:
        model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1] 
    return {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test ROC-AUC': roc_auc_score(y_test, y_test_proba),
        'Overfit': accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred)
    }