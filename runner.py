import os
import sys
import model
import utils
import transfer
import numpy as np

if __name__ == '__main__':

    if sys.argv[2] == 'dry':

        os.chdir(os.path.dirname(sys.argv[1]))
        train = utils.load_data_to_memory(os.path.join(sys.argv[1], 'train'))
        test = utils.load_data_to_memory(os.path.join(sys.argv[1], 'test'))
        train_x = model.get_data(train, 224)
        train_y = model.get_labels(train)
        test_x = model.get_data(test, 224)
        test_y = model.get_labels(test)
        model_ = model.get_MobileNet_v2((224,224,1))
        train_x = np.expand_dims(train_x, -1)
        test_x = np.expand_dims(test_x, -1)
        model_ = model.build(train_x, train_y, test_x, test_y, model_)

    elif sys.argv[2] == 'transfer':
        fer_dataset = '~/Downloads/fer2013/fer2013.csv'
        os.chdir(os.path.dirname(sys.argv[1]))
        train = utils.load_data_to_memory(os.path.join(sys.argv[1], 'train'))
        test = utils.load_data_to_memory(os.path.join(sys.argv[1], 'test'))
        train_x = model.get_data(train, 48)
        train_y = model.get_labels(train)
        test_x = model.get_data(test, 48)
        test_y = model.get_labels(test)
        transfer_data_path = fer_dataset
        ex_img, ex_y = transfer.get_ex_data(transfer_data_path)

        train_x = np.append(train_x, ex_img, axis=0)
        train_y = np.append(train_y, ex_y, axis=0)
        model_ = model.get_MobileNet_v2((48,48,1))
        train_x = np.expand_dims(train_x, -1)
        test_x = np.expand_dims(test_x, -1)
        model_ = model.build(train_x, train_y, test_x, test_y, model_)


    else:
        print('please use "dry" or "transfer" as the second argument when running runner.py')