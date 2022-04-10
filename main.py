import numpy as np
import pandas as pd


def Build_Data(size: int, Part):
    class_array = np.zeros(size)
    data_x = np.zeros(size)
    data_y = np.zeros(size)
    for i in range(size):
        m = np.random.randint(-10000, 10000)
        n = np.random.randint(-10000, 10000)
        data_x[i] = round((m / 100), 2)
        data_y[i] = round((n / 100), 2)

        if Part == 'A':
            if n / 100 <= 1:
                class_array[i] = -1
            else:
                class_array[i] = 1
    x = data_x.astype(np.float64)
    y = data_y.astype(np.float64)
    classify = class_array.astype(np.float64)
    return x, y, classify


if __name__ == '__main__':
    x, y, c = Build_Data(1000, 'A')
    print(x)
