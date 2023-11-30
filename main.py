from functions import *

if __name__ == '__main__':
    file_path = "data//poseangles.csv"
    X_train, X_test, y_train, y_test = load_data(file_path)

    # choices are: rfc, svm, knn
    classifier_model = "svc"
    show_csv = True
    if classifier_model == "rfc":
        randomforestclassifier(X_train, X_test, y_train, y_test, show_csv)
    elif classifier_model == "svc":
        supportvectorvclassifier(X_train, X_test, y_train, y_test, show_csv)
    elif classifier_model == "knn":
        kneighborsclassifier(X_train, X_test, y_train, y_test, show_csv)

    '''
    best one looks like its SVC, with accuracy of 0.9694835680751174
    '''
