from model_rej.model_inf import ABCNet_inf
from model_rej.configs import get_cfg
import torch
import cv2
import pickle
import timeit


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print('Code block' + self.name + ' took: ' + str(self.took) + ' ms')

def mobile_net_svm_rejector(clf, abcnet, image, device = 'cpu'):

    """
    :param image: cropped image ndarray 3 channel if size below 200 * 200 assert error
    :param device: using cuda or cpu
    :return: 1/0 text/non text
    """
    #load a trained svm classifier
    # with CodeTimer('Load SVM'):
    #     clf = pickle.load(open('svm_layer_1.p', 'rb'))
    if image.shape[0] <= 200 or image.shape[1] <= 200:
        raise ValueError('Image size must larger than 200 * 200')

    #extract cropped images feature
    with torch.no_grad():
        # with CodeTimer('Load ABCNET'):
        #     cfg = get_cfg()
        #     abcnet = ABCNet_inf(cfg, device, output_stage=0)
        #     abcnet.to(device)
        #     abcnet.eval()
        #     model_path = 'model_rej/model_mobile_all_bn_new.pth'
        #     abcnet.load_state_dict(torch.load(model_path, map_location='cpu')['model'], strict=False)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img = img.unsqueeze(0)
        # with CodeTimer('Extract Features'):
        features = abcnet(img)
        if device != 'cpu':
            features = features[0].cpu().numpy()
        else:
            features = features[0].numpy()
        features = features.flatten()
    # with CodeTimer('SVM predict'):
    predict = clf.predict([features])

    return predict

