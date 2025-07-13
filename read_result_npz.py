import numpy as np

def read_result_npz(file_path):
    data = np.load(file_path)
    return data

if __name__ == '__main__':
    file_path = "C:\\Users\\60353\Downloads\Documents\LSTM_lr0.01_wd0.0_up_sample_timely_fold6_result.npz"
    data = read_result_npz(file_path)
    print(data.files)
    for key in data.files:
        print(key, data[key].tolist())