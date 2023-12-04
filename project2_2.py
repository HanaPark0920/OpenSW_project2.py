import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error
import numpy as np


def sort_dataset(dataset_df):
    """
    함수 설명
    : 주어진 데이터를 year을 기준으로 오름차순으로 정렬하는 함수

    parameter
    : dataset_df

    return
    : sort_df, 정렬된 데이터
    """
    sorted_df = dataset_df.sort_values(by='year')
    return sorted_df


def split_dataset(dataset_df):
    """
    함수 설명
    : 'salary'열은 레이블로 사용.
        레이블 값은 0.001을 곱해서 다시 스케일링 수행.
        train dataset -> index range of [:1718]
        test dataset -> index range of [1718:]의 범위만큼을 사용.

    parameter
    : dataset_df

    return
    : n X_train, X_test, Y_train, Y_test dataframes
    """
    # 'salary'열 값에 0.001을 곱한 후 다시 스케일링 수행.
    dataset_df['salary'] *= 0.001

    # 인덱스 범위 나누기
    train_df = dataset_df.iloc[:1718]
    test_df = dataset_df.iloc[1718:]

    # x 와 y로 분리하기
    X_train = train_df.drop(['salary'], axis=1) # axis=1을 이용해서 'salary' 열 제외함.
    Y_train = train_df['salary']

    X_test = test_df.drop(['salary'], axis=1)
    Y_test = test_df['salary']

    return X_train, X_test, Y_train, Y_test


def extract_numerical_cols(dataset_df):
    """
        함수 설명
        : dataset_df에서 숫자로 된 특성만 추출을 한 데이터 프레임을 리턴하는 함수
        Numerical columns: 'age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR',
        'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war’

        parameter
        : dataset_df

        return
        : numerical_data: 숫자형 열만을 포함한 df
        """
    # 숫자로된 열 목록들
    numerical_col = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR',
                         'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']

    #숫자로된 열만 추출해서 새로운 df 생성하기
    numerical_data = dataset_df[numerical_col]

    return numerical_data



def train_predict_decision_tree(X_train, Y_train, X_test):
    """
        함수 설명
        : 주어진 X_train과 Y_train을 사용해서 decision tree regressor model을 훈련하는 함수
            이후 trained model을 사용해서 X_test에 대한 예측 결과를 리턴

        import ('from sklearn.tree import DecisionTreeRegressor')
        : DecisionTreeRegressor : decision tree regressor model을 구현한 걸 사용하기 위해 가져옴

        parameter
        : X_train, Y_train, X_test

        return
        :  prediction result of X_test by using the trained model
        """
    # decision tree regressor model 훈련하기
    dt_model = DecisionTreeRegressor()
    dt_model.fit(X_train, Y_train)

    # X_test에 대한 예측 수행하기
    dt_predictions = dt_model.predict(X_test)

    return dt_predictions


def train_predict_random_forest(X_train, Y_train, X_test):
    """
        함수 설명
        : 주어진 X_train과 Y_train을 사용해서 random forest regressor model 을 훈련하는 함수
            이후 trained model을 사용해서 X_test에 대한 예측 결과를 리턴

        import ('from sklearn.ensemble import RandomForestRegressor')
        : RandomForestRegressor : random forest regressor model을 구현한 걸 사용하기 위해 가져옴

        parameter
        : X_train, Y_train, X_test

        return
        :  prediction result of X_test by using the trained model
        """
    # random forest regressor model 훈련하기
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, Y_train)

    # X_test에 대한 예측 수행하기
    rf_predictions = rf_model.predict(X_test)

    return rf_predictions


def train_predict_svm(X_train, Y_train, X_test):
    """
        함수 설명
        : 주어진 X_train과 Y_train을 사용하여
        pipeline consists of a standard scaler and SVM model을 훈련시키는 함수

        import('from sklearn.svm import SVR', 'from sklearn.preprocessing import StandardScaler', 'from sklearn.pipeline import make_pipeline')
        : svm 모델과 관련된 전처리 도구를 사용하기 위해서 불러옴

        parameter
        : X_train, Y_train, X_test

        return
        : prediction result of X_test by using the trained model
        """
    # standard scaler와 svm_model로 구성된 pipeline을 생성하기
    # make_pipeline 함수를 사용해서 pipeline 생성
    svm_model = make_pipeline(StandardScaler(), SVR())

    # pipeline 훈련하기
    svm_model.fit(X_train, Y_train)

    # X_test에 대한 예측 수행하기
    svm_predictions = svm_model.predict(X_test)

    return svm_predictions



def calculate_RMSE(labels, predictions):
    """
        함수 설명
        : 주어진 labels(기존 레이블), predictions(예측된 레이블)을 사용해서 RMSE 계산 후 리턴

        import ('from sklearn.metrics import mean_squared_error', 'import numpy as np')
        : RMSE 계산을 위해 scikit-learn의 mean_squared_error 함수와 numpy 사용

        parameter
        : labels, predictions

        return
        : Calculate and return RMSE using given labels and predictions
        """

    # RMSE 계산하기
    # mean_squared_error 함수를 사용하여 labels과 predictions 사이의 평균 제곱 오차를 계산
    RMSE = np.sqrt(mean_squared_error(labels, predictions))

    return RMSE



if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))