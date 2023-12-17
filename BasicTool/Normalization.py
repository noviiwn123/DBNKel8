import pandas as pd


def normalizeDataset(dataset):
    print(dataset.min(), dataset.max())
    normalize = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    # normalize = (dataset - 1) / (10 - 1)
    return normalize


if __name__ == '__main__':
    file_path = 'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/dataset/TestData.csv'
    # file_path = 'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/dataset/TrainData.csv'
    dataset = pd.read_csv(file_path)
    dataset = normalizeDataset(dataset)
    print(dataset.head(5))
    path = 'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/normalizedData/TestData.csv'
    # path = 'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/normalizedData/TrainData.csv'
    dataset.to_csv(path, header=True, index=False)
