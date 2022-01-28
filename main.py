import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

wheel_radius_cm = 5.1/2

def main():
    data_frame = pd.read_csv('data/Lab13Export.csv', sep=',', header=0)
    generate_graph(1, 2, data_frame)

def generate_graph(index_1, index_2, data_frame):
    # Get Headers
    headers = data_frame.columns.values
    plt.plot(data_frame[headers[index_1]], data_frame[headers[index_2]], 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Dependent Variable')
    plt.title('Plot of 1 vs 2')
    plt.show()



if __name__ == '__main__':
    main()