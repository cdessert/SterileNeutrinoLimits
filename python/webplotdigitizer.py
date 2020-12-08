import numpy as np
import pandas as pd

def read_csv(filename):
	'''Takes the output data format of webplotdigitizer and returns the x and y python arrays'''

	# Assumes the data is listed in the file filename
	df = pd.read_csv(filename,header=None)
	x = np.array(df.transpose().iloc[0])
	y = np.array(df.transpose().iloc[1])

	return x,y