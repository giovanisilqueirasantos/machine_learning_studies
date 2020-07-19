import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class WineDataset():
    """
    The Wine dataset is a open-source dataset that is available from the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/Wine); it consists of 178 wine samples with 13 features describing their different chemical properties.
    """

    def __init__(self, test_size, random_state):
        self.df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
        self.df.columns = [
            'Class label',
            'Alcohol',
            'Malic acid',
            'Ash',
            'Alcalinity of ash',
            'Magnesium',
            'Total phenols',
            'Flavanoids',
            'Nonflavanoid phenols',
            'Proanthocyanins',
            'Color intensity', 
            'Hue',
            'OD280/OD315 of diluted wines',
            'Proline'
        ]
        X, y = self.df.iloc[:, 1:].values, self.df.iloc[: , 0].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        mms = MinMaxScaler()
        stdsc = StandardScaler()
        self.X_norm_train = mms.fit_transform(self.X_train)
        self.X_norm_test = mms.transform(self.X_test)
        self.X_std_train = stdsc.fit_transform(self.X_train)
        self.X_std_test = stdsc.transform(self.X_test)