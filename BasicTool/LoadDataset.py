import pandas as pd


def loadDataset(file_path):
    dataframe = pd.read_csv(file_path)
    target = dataframe.loc[:, dataframe.columns == 'Class']
    dataset = dataframe.loc[:, dataframe.columns != 'Class']

    return dataset, target


if __name__ == '__main__':
    filename = 'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/dataset/TrainData.csv'
    dataset, target = loadDataset(filename)
    print(dataset.head(5))
    print(target.head(5))