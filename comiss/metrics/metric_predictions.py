from typing import Any
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, balanced_accuracy_score,roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.metrics import r2_score, mean_tweedie_deviance

import numpy as np

# ALL CLASSIFICATION METRICS HAS (y_pre, y_true) order!!
class OverallAccuracy():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(y_pred == y_true)
	
class AverageAccuracy():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return balanced_accuracy_score(y_true, y_pred)
	
class F1Score():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		return f1_score(y_true, y_pred, average=self.average)

class Precision():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
	    return precision_score(y_true, y_pred, average=self.average)
    
class Recall():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
	    return recall_score(y_true, y_pred, average=self.average)
    
class Kappa():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return cohen_kappa_score(y_true, y_pred)
	
class MCC():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return matthews_corrcoef(y_true, y_pred)
	
class ConfusionMatrix():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return confusion_matrix(y_true, y_pred)
	
class Get_Data():
	def __init__(self, which="true"):
		self.which = which

	def __call__(self, y_pred,y_true):
		data_check = y_true if self.which == "true" else y_pred
		labels_n = np.unique(y_true)
		return [np.sum(data_check==v) for v in labels_n]

#METRICS FOR GIVING PROBABILITIES AS OUTPUT
class ROC_AUC():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) > 1 and np.max(y_true) == 1:
			y_pred = y_pred[:,1]
		return roc_auc_score(y_true, y_pred, average=self.average, multi_class="ovr")
	
class mAP():
	def __init__(self, average):
		self.average = None if average=="none" else average
	def __call__(self, y_pred, y_true):
		return average_precision_score(y_true, y_pred, average=self.average)
	
class CatEntropy(): #normalized
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) > 1:
			K = y_pred.shape[1]
			y_pred = np.clip(y_pred, 1e-10, 1.0)
			entropy = - (y_pred * np.log(y_pred)).sum(axis=-1) / np.log(K)
			return entropy.mean(axis=0)
    
class P_max():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) == 2:
			N, K = y_pred.shape
			p_max_x =  np.max(y_pred, axis =-1)
			return p_max_x.mean(axis=0)
    
class LogP(): #un-normalized
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) == 2:
			N, K = y_pred.shape
			y_pred = np.clip(y_pred, 1e-10, 1.0)
			return ( np.log(y_pred[np.arange(N), y_true]) ).mean(axis=0)
    
class P_true():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		if len(y_pred.shape) == 2:
			N, K = y_pred.shape
			return ( y_pred[np.arange(N), y_true] ).mean(axis=0)


class R2Score():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return r2_score(y_true, y_pred)
	
class MAE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(np.abs(y_true -y_pred))
	
class MedAE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.median(np.abs(y_true -y_pred))
	
class RMSE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.sqrt(np.mean((y_true - y_pred)**2))
	
class rRMSE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.sqrt(np.mean((y_true - y_pred)**2/np.mean(y_true)))
	
class MAPE():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(np.abs((y_true - y_pred)/y_true))
	
class BIAS():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return np.mean(y_pred - y_true)
	
class TweediePoisson():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		return mean_tweedie_deviance(y_true, y_pred, power=1)
	
class PCorr():
	def __init__(self):
		pass
	def __call__(self, y_pred, y_true):
		y_pred_mean = y_pred.mean()
		y_pred_std = y_pred.std()
		y_true_mean = y_true.mean()
		y_true_std = y_true.std()

		covariance_ = np.mean((y_pred-y_pred_mean)*(y_true-y_true_mean))

		return covariance_/(y_pred_std*y_true_std)