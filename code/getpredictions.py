import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_label(label):
    labelDictionary = {'plastic': 1, 'other': 2}
    for name in labelDictionary:
        if label == labelDictionary[name]:
            label = name
        return label


def getprediction(model, img, raw_image, device):
    boxes_array = []
    label_array = []
    model.eval()
    image = raw_image
    with torch.no_grad():
        '''
        prediction Like:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
    prediction = model([img.to(device)])
    scores = prediction[0]['scores']
    boxes_array = prediction[0]['boxes']
    boxes_array = boxes_array[scores > 0.5]
    label_array = prediction[0]['labels']
    label_array = label_array[scores > 0.5]
    label_name_array = []

    for index in range(len(boxes_array)):
        xmin = round(np.int(boxes_array[index, 0]))
        ymin = round(np.int(boxes_array[index, 1]))
        xmax = round(np.int(boxes_array[index, 2]))
        ymax = round(np.int(boxes_array[index, 3]))

        score = round(scores[index].item(), 2)
        label = label_array[index]
        label = get_label(label)
        label_name_array.append(label)

        boxtext = str(label) + " " + str(score)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (51, 51, 255), thickness=1)
        cv2.putText(image, boxtext, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), thickness=1)

        plt.figure(figsize=(20, 15))
        plt.imshow(image)
        return prediction