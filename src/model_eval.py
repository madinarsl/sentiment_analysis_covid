from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    '''
    Evaluates a model on train and test sets. 
    Returns a dictionary with metrics (Accuracy, ROC-AUC, F1-Score) 
    '''
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1] 

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1] 
    return {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Train F1-Score': f1_score(y_train, y_train_pred),
        'Test F1-Score': f1_score(y_test, y_test_pred),
        'Train ROC-AUC': roc_auc_score(y_train, y_train_proba),
        'Test ROC-AUC': roc_auc_score(y_test, y_test_proba)
    }