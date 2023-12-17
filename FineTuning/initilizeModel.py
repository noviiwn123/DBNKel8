import pandas as pd
import pickle


def init():
    paths = ['C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/weights/RBM1.csv',
             'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/weights/RBM2.csv',
             'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/weights/RBM3.csv',
             'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/weights/RBM4.csv']

    network = list()
    for path in paths:
        df = pd.read_csv(path)
        dframe = df.drop(columns='hidden0')
        # print(dframe)

        neurons = list()
        for i in range(len(dframe.columns)):
            neuron = dframe.iloc[:, i]
            # print(neuron)
            neurons.append({'weights': list(neuron)})

        # print(neurons)
        network.append(neurons)

    # print(network)
    return network


if __name__ == '__main__':
    network = init()
    # for row in network:
    #     print(row)
    filename = 'C:/Users/Wahyu Novitasari/Downloads/SEMESTER 7/Machine Learning/DBN/classifier/experiments/preTrainedModel_05.pkl'
    pickle.dump(network, open(filename, 'wb'))
    print(network)