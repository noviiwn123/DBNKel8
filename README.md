# Implementasi Deep Belief Network (DBN) sebagai Algoritma untuk Mendeteksi Kanker Payudara 

Dalam proyek ini, kami menggunakan Deep belief network (DBN) untuk melatih model. Kemudian, kami menggunakan algoritma back-propagation untuk mengoptimalkan model. Proyek ini sepenuhnya dikembangkan dari awal menggunakan library dasar seperti numpy, pandas, dan matplotlib.
Dataset diambil dari Repositori Pembelajaran Mesin UCI  

Link to dataset: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

DETAIL FOLDER

	1. Basic Tool
		-> LoadDataset.py ==> load dataset 
 		-> Normalization.py ==> normalize dataset 


	2. classifier
	In this folder trained classifier model is saved.
 
	3. dataset 
	Terdapat 4 file CSV di dalam folder ini. 
	The original.csv is original dataset from winsconsin machinelearning repositiory.
		-> csv_db.csv ==> cleaned datasets. It is divided into train and test dataset. (TrainData.csv and TestData.csv)
		-> TrainData.csv ==> for training classifier model
		-> TestData.csv ==> for testing trained classifier model

	4. normalization
	Dataset disimpan ke dalam folder ini setelah normalisasi.
		-> normalized.csv
		-> TestData.csv
		-> TrainData.csv

	5. Preprocessing
	Implementasi Deep Belief Network (DBN)
		-> Activation.py ==> activation function
		-> DeepBeliefNet.py ==> creating deep belief network 
		-> InitilizationRBM.py ==> initilizating RBM (Restricted Boltzmann Machine)
		-> TrainRBM.py ===> training RBM without creating batch of dataset
		-> TrainRBMTest.py ===> training RBM by creating batch of dataset

	Note: network layers 
  
		Input layer = 9 neurons (8 features and 1 bias)
		1st hidden layer => 25 neurons
		2nd hidden layer => 25 neurons
		3rd hidden layer => 10 neurons
		output layer => 1 neuron


	6. python
	this folder is created to provide user service using UI. I used web based UI but I have not mentioned here. 
	two folders: 
		1. classifier 
			-> fineTunedModel.pkl	 ===> trained model to classify new data’s

		2. json 	
			-> data.json ==> data extracted from user
			-> result.json ==> result after classification
	7. weights
	weights from Pre-Training 
		-> initialWeight.csv
		-> RBM1.csv ==> weights between input and 1st hidden layer
		-> RBM2.csv ==> weight between 1st and 2nd hidden layer
		-> RBM3.csv ==> weight between 2nd and 3rd hidden layer 
		-> RBM4.csv ==> weight between 3rd and output layer
