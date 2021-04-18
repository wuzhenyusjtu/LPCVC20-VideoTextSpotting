import pickle
import numpy as np
import sys
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.append(parentdir)


def testSVM(model, test_data, test_pth):
    cnt_correct = 0
    for i, file in enumerate(test_data):
        x = np.load(os.path.join(test_pth, file))
        predict_label = model.predict([x])
        if file[-5] == '1':
            true_label = 1
        elif file[-5] == '0':
            true_label = 0
        else:
            print('Error')
            break

        if predict_label == true_label:
            cnt_correct += 1

    print('Accuracy for SVM is {}'.format(cnt_correct / len(test_data)))


if __name__ == '__main__':
    ''' 
        # Parse the arguments
        # Some useful parameters:
        # trainRoot: Path to training images, which are generated using FOTS model. Image with bounding boxes detected 
        # are marked as 1, otherwise marked as 0.
        # testRoot: Path to testing images.
        # pretrained: Path to pretrained quantized part one for FOTS model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainRoot', default='data/train/', help='path to dataset for training')
    parser.add_argument('--testRoot', default='data/test/', help='path to dataset for testing')
    parser.add_argument('--save-dir', required=True, help='path to save models')

    opt = parser.parse_args()
    train_dir = opt.trainRoot
    test_dir = opt.testRoot

    train_list = [file for file in os.listdir(train_dir) if file.endswith('.npy')]
    test_list = [file for file in os.listdir(test_dir) if file.endswith('.npy')]

    train_set = np.zeros((len(train_list), 128))
    train_labels = np.zeros(len(train_list))

    for i, file in enumerate(train_list):
        x = np.load(os.path.join(train_dir, file))
        train_set[i] = x
        if file[-5] == '1':
            train_labels[i] = 1
        elif file[-5] == '0':
            train_labels[i] = 0
        else:
            print('Wrong naming dataset')
            break

    print(train_set.shape)
    print(train_labels.shape)

    # LinearSVC
    clf = make_pipeline(StandardScaler(), svm.LinearSVC(random_state=0, tol=1e-5))
    clf.fit(train_set, train_labels)
    pred = clf.predict(train_set)
    print('\nTraining Accuracy:', accuracy_score(train_labels, pred))
    filename = '{}/svm_linearSVC.sav'.format(opt.save_dir)
    pickle.dump(clf, open(filename, 'wb'))
    testSVM(clf, test_list, test_dir)

    # rbf Kernel
    C = [0.01, 0.1, 1, 10, 100]
    Gamma = [0.01, 0.1, 1, 10, 100]
    for c in C:
        for gamma in Gamma:
            clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            clf.fit(train_set, train_labels)
            pred = clf.predict(train_set)
            print('\nTraining Accuracy:', accuracy_score(train_labels, pred))
            filename = '{}/svm_C_{}_gamma_{}.sav'.format(opt.save_dir, c, gamma)
            pickle.dump(clf, open(filename, 'wb'))
            testSVM(clf, test_list, test_dir)
