# Get json files from train.txt and test.txt from MJ Dataset
import os
import random
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, help='path to dataset')
    parser.add_argument('--train_num', required=True, help='train data number')
    parser.add_argument('--test_num', required=True, help='test data number')
    opt = parser.parse_args()
    
    mjdata_dir = opt.datadir

    img_pathlist_train = []
    label_list_train = []
    cnt = 0
    with open(os.path.join(mjdata_dir,'annotation_train.txt'), 'r', encoding='ascii') as f:
        line = f.readline() 
        while line: 

            lines = line.split(' ')
            t = random.randint(0, 100)
            if t <= 50:
                line = f.readline() 
                continue
            cnt += 1
            if cnt >= int(opt.train_num):
                break
            img_pathlist_train.append(lines[0][2:])

            line = f.readline() 
            
    img_pathlist_test = []
    cnt = 0
    with open(os.path.join(mjdata_dir,'annotation_test.txt'), 'r', encoding='ascii') as f:
        line = f.readline() 
        while line: 

            lines = line.split(' ')
            t = random.randint(0, 100)
            if t <= 50:
                line = f.readline() 
                continue
            cnt += 1
            if cnt >= int(opt.test_num):
                break
            img_pathlist_test.append(lines[0][2:])
            line = f.readline() 

    with open(os.path.join(opt.datadir, 'test_new.json'), 'w') as file:
        json.dump(img_pathlist_test, file)
    with open(os.path.join(opt.datadir,'train_new.json'), 'w') as file:
        json.dump(img_pathlist_train, file)
if __name__ == '__main__':
    main()
