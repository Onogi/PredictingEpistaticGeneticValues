from __future__ import print_function
import numpy as np
import pandas as pd
import os
import re
import glob
import keras
import h5py
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.metrics import mean_squared_error

def RunMLP (DataNo, SimNo_start, SimNo_end, ModelNo, Nloci, Nunit, Naddlayer, Rdrop, Sbatch, Nepoch, Vsplit, AddBN = False, Bias = False):
	#when Naddlayer=0, a NN with one hidden layer is build
	#NN
	Model = Sequential()
	Model.add(Dense(Nunit, activation='relu', input_shape=(Nloci,), use_bias=Bias))
	if Rdrop > 0.0:
		Model.add(Dropout(Rdrop))
	if AddBN:
		Model.add(BatchNormalization())
	if Naddlayer > 0:
		for i in range(Naddlayer):
			Model.add(Dense(Nunit, activation='relu', use_bias=Bias))
			if Rdrop > 0.0:
				Model.add(Dropout(Rdrop))
			if AddBN:
				Model.add(BatchNormalization())
	Model.add(Dense(1, activation='linear', use_bias=Bias))
	if AddBN:
		Model.add(BatchNormalization())
	Model.compile(loss = 'mean_squared_error', optimizer='Adam', metrics=['mse'])

	for sim in range(SimNo_start, SimNo_end):
		#read genotypes
		Geno_train = pd.read_csv(filepath_or_buffer="Data%d.sim%d.Geno.train.csv"%(DataNo, sim+1), sep=",")
		Geno_train = np.delete(Geno_train.values, 0, 1)#delete row numbers
		Geno_train = Geno_train.astype('float')

		#read phenotypes
		Pheno_train = pd.read_csv(filepath_or_buffer="Data%d.sim%d.Pheno.train.csv"%(DataNo, sim+1), sep=",")
		Pheno_train = Pheno_train.values[:,1]#use phenotypic values. Specify 2 when genotypic values are used
		Pheno_train_copy = np.copy(Pheno_train)
		Pheno_train = (Pheno_train - Pheno_train_copy.mean()) / Pheno_train_copy.std()

		#Train
		Path = 'Data%d.sim%d.model%d-{epoch:02d}-{mean_squared_error:.8f}-{val_mean_squared_error:.8f}-.hdf5'%(DataNo, sim+1, ModelNo)
		Cb = ModelCheckpoint(filepath=Path, monitor='val_mean_squared_error', save_best_only=True, save_weights_only=False, period=10)
		Result = Model.fit(Geno_train, Pheno_train, batch_size=Sbatch, epochs=Nepoch, validation_split=Vsplit, callbacks=[Cb])

		#Remove files
		Weightfiles = glob.glob('Data%d.sim%d.model%d-*.hdf5'%(DataNo,sim+1,ModelNo))
		for i in range(len(Weightfiles) - 1):#the last one is the best one
			os.remove(Weightfiles[i])


def RetrieveMLP (DataNo, SimNo_start, SimNo_end, ModelNo):

	#retrieve results
	Metrics = np.zeros(shape=(SimNo_end-SimNo_start,7))

	for sim in range(SimNo_start, SimNo_end):
		#read genotype files
		Geno_train = pd.read_csv(filepath_or_buffer="Data%d.sim%d.Geno.train.csv"%(DataNo,sim+1), sep=",")
		Geno_train = np.delete(Geno_train.values, 0, 1)#delete row numbers
		Geno_train = Geno_train.astype('float')

		Geno_test = pd.read_csv(filepath_or_buffer="Data%d.sim%d.Geno.test.csv"%(DataNo,sim+1), sep=",")
		Geno_test = np.delete(Geno_test.values, 0, 1)
		Geno_test = Geno_test.astype('float')

		#read BVs
		Pheno_train = pd.read_csv(filepath_or_buffer="Data%d.sim%d.Pheno.train.csv"%(DataNo,sim+1), sep=",")
		Pheno_train_copy = np.copy(Pheno_train.values[:,1])#to scale back
		Pheno_train = Pheno_train.values[:,2]#specify 1 when phenotypic values are used
		Pheno_train = np.reshape(Pheno_train,(1,Pheno_train.shape[0]))

		Pheno_test = pd.read_csv(filepath_or_buffer="Data%d.sim%d.Pheno.test.csv"%(DataNo,sim+1), sep=",")
		Pheno_test = Pheno_test.values[:,2]
		Pheno_test = np.reshape(Pheno_test,(1,Pheno_test.shape[0]))

		#read hd5 file
		Weightfile = glob.glob('Data%d.sim%d.model%d-*.hdf5'%(DataNo,sim+1,ModelNo))

		#load model
		Model = load_model(Weightfile[0], compile=False)

		#predict
		EBV_train = Model.predict(Geno_train, batch_size=Geno_train.shape[0])
		EBV_train = EBV_train * Pheno_train_copy.std() + Pheno_train_copy.mean()
		EBV_train = np.reshape(EBV_train,(1,EBV_train.shape[0]))
		Cor_train = np.corrcoef(EBV_train,Pheno_train)[0,1]
		Coef_train = np.poly1d(np.polyfit(EBV_train.flatten(), Pheno_train.flatten(), 1))
		MSE_train = mean_squared_error(EBV_train.flatten(), Pheno_train.flatten())

		EBV_test = Model.predict(Geno_test, batch_size=Geno_test.shape[0])
		EBV_test = EBV_test * Pheno_train_copy.std() + Pheno_train_copy.mean()
		EBV_test = np.reshape(EBV_test,(1,EBV_test.shape[0]))
		Cor_test = np.corrcoef(EBV_test,Pheno_test)[0,1]
		Coef_test = np.poly1d(np.polyfit(EBV_test.flatten(), Pheno_test.flatten(), 1))
		MSE_test = mean_squared_error(EBV_test.flatten(), Pheno_test.flatten())

		#stack
		MSE = re.split('-', Weightfile[0])
		Metrics[sim,:]=[float(MSE[1]),MSE_train,Cor_train,float(Coef_train[1]),MSE_test, Cor_test, float(Coef_test[1])]

	Result = pd.DataFrame(Metrics, columns=('BestEpoch','Mse_train','Cor_train','Coef_train','MSE_test','Cor_test','Coef_test'))
	Result.to_csv('Metrics.Data%d.sim%d-%d.model%d.csv'%(DataNo,SimNo_start+1,SimNo_end,ModelNo))


def RetrieveWeights (DataNo, SimNo_start, SimNo_end, ModelNo):

	Nlayer = (ModelNo%100)//10 + 2

	for sim in range(SimNo_start, SimNo_end):
		Weightfile = glob.glob('Data%d.sim%d.model%d-*.hdf5'%(DataNo,sim+1,ModelNo))
		h5file = h5py.File(Weightfile[0],'r')

		for layer in range(Nlayer):
			Which = list(h5file['model_weights'])[layer + Nlayer]
			Weights = h5file['model_weights/%s/%s/kernel:0'%(Which,Which)]
			Value = pd.DataFrame(Weights[()])
			Value.to_csv('Weights.Data%d.sim%d.model%d.%s.csv'%(DataNo,sim+1,ModelNo,Which))


