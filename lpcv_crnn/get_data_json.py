# Get json files from train.txt and test.txt from MJ Dataset
import os
import random
import argparse
import json

'''
Get new train/test lists from original one

Args:
    txt_file: path to the annotation txt file of the original mjdataset
    num: number of trainging/test images you want to use
    prob: probability to continue
    
Return:
    A list contains the paths to the images/data you select. 
    The length of the list is the same with the number parameter.
'''
def get_selected_list(txt_file, num, prob):
    path_list = []
    cnt = 0
    with open(txt_file, 'r', encoding='ascii') as f:
        line = f.readline() 
        while line: 
            lines = line.split(' ')
            t = random.randint(0, 100)
            if t <= 100*prob:
                line = f.readline() 
                continue
            cnt += 1
            if cnt >= num:
                break
            path_list.append(lines[0][2:])

            line = f.readline() 
    return path_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, help='path to dataset')
    parser.add_argument('--train_num', required=True, help='train data number')
    parser.add_argument('--test_num', required=True, help='test data number')
    parser.add_argument('--prob', default=0.5, help='probability for data selection method to continue')
    opt = parser.parse_args()
        
    img_pathlist_train = get_selected_list(os.path.join(opt.datadir,'annotation_train.txt'), int(opt.train_num), float(opt.prob))
    img_pathlist_test = get_selected_list(os.path.join(opt.datadir,'annotation_test.txt'), int(opt.test_num), float(opt.prob))
#     print(len(img_pathlist_train))
#     print(len(img_pathlist_test))
    with open(os.path.join(opt.datadir,'train_new.json'), 'w') as file:
        json.dump(img_pathlist_train, file)
    with open(os.path.join(opt.datadir, 'test_new.json'), 'w') as file:
        json.dump(img_pathlist_test, file)

if __name__ == '__main__':
    main()
