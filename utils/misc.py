def trace(func):
    def wrapper():
        print(func.__name__, "begins")
        func()
        print(func.__name__, "ends")
        return
    return wrapper

import numpy as np
def isnan_ds(dataset):
    for i in range(len(dataset)):
        isnan = np.sum(np.isnan(dataset[i]['timeseries'].numpy()))
        #print(isnan)
        if isnan>0:
            print(dataset[i]['id'])