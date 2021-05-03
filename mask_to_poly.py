
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

def mask_to_poly(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    return contours

def IoU_contours(contour_pred, contour_label, label_size):

    true_pos = 0
    false_pos = 0
    num_true_build = len(contour_label)
    for i in range(0,len(contour_pred)):
        img1 = cv2.drawContours(np.zeros(label_size), contour_pred, i, 1, cv2.FILLED)
        IoU_max = -1
        max_index = -1
        for j in range(0,len(contour_label)):
            img2 = cv2.drawContours(np.zeros(label_size), contour_label, j, 1, cv2.FILLED)

            inter = np.logical_and(img1, img2)
            inter_sum = np.sum(inter)

            union = np.logical_or(img1, img2)
            union_sum = np.sum(union)

            if union_sum == 0:
                IoU = -1
            else:
                IoU = inter_sum / union_sum

            if IoU > IoU_max:
                IoU_max = IoU
                max_index = j

        if IoU_max >= 0.5:
            true_pos += 1
            del contour_label[max_index]
        else:
            false_pos += 1

    false_neg = num_true_build - true_pos
    return true_pos, false_pos, false_neg
    #precision = true_pos / (true_pos + false_pos)
    #recall = true_pos / (true_pos + false_neg)


def IoU_masks(mask_pred, mask_label):
    cont_pred = mask_to_poly(mask_pred)
    cont_label = mask_to_poly(mask_label)

    true_positives, false_positives, false_negatives = IoU_contours(cont_pred, cont_label, np.shape(mask_label))

    return true_positives, false_positives, false_negatives


def measures(tp, fp, fn):
    if (tp + fp) == 0:
        print("Error: Precision (tp + fp) == 0")
        precision = np.nan
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        print("Error: Recall (tp + fn) == 0")
        recall = np.nan
    else:
        recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


if __name__ == '__main__':

    img_dir = '/SN1_buildings_train_AOI_1_Rio_3band/3band/'
    label_dir = '/building_mask/'

    # img_path = os.getcwd() + img_dir + '3band_AOI_1_RIO_img163.tif'
    # image = np.array(Image.open(img_path))[:406,:438]

    lbl_path = os.getcwd() + label_dir + '3band_AOI_1_RIO_img1474.tif'
    label1 = np.array(Image.open(lbl_path))[:406,:438]

    lbl_path = os.getcwd() + label_dir + '3band_AOI_1_RIO_img1475.tif'
    label2 = np.array(Image.open(lbl_path))[:406,:438]

    lbl_path = os.getcwd() + label_dir + '3band_AOI_1_RIO_img1.tif'
    label3 = np.array(Image.open(lbl_path))[:406,:438]

    #TESTING STARTS HERE
    cont_test_1 = mask_to_poly(label1)
    img_test = cv2.drawContours(np.zeros(np.shape(label1)), cont_test_1, -1, 255)
    cv2.imwrite('test1.png', img_test)

    cont_test_2 = mask_to_poly(label2)
    img_test = cv2.drawContours(np.zeros(np.shape(label2)), cont_test_2, -1, 255)
    cv2.imwrite('test2.png', img_test)

    cont_test_3 = mask_to_poly(label3)
    img_test = cv2.drawContours(np.zeros(np.shape(label3)), cont_test_3, -1, 255)
    cv2.imwrite('test3.png', img_test)


    # Test same labels to get accuracy of 1.0
    true_pos, false_pos, false_neg = IoU_masks(label1, label1)
    print("Num of buildings: ", len(cont_test_1))
    print("Num of predicted Buildings: ", len(cont_test_1))
    print("TP, FP, FN: ")
    print(true_pos, false_pos, false_neg)
    prec, rec, f1 = measures(true_pos, false_pos, false_neg)
    print("Prec, Rec, F1: ")
    print(prec, rec, f1)

    # Test different labels to get low accuracy
    true_pos, false_pos, false_neg = IoU_masks(label1, label2)
    print("\nNum of buildings: ", len(cont_test_2))
    print("Num of predicted Buildings: ", len(cont_test_1))
    print("TP, FP, FN: ")
    print(true_pos, false_pos, false_neg)
    prec, rec, f1 = measures(true_pos, false_pos, false_neg)
    print("Prec, Rec, F1: ")
    print(prec, rec, f1)

    # Test 98 real buildins but 0 predicted
    true_pos, false_pos, false_neg = IoU_masks(label1, label3)
    print("\nNum of buildings: ", len(cont_test_3))
    print("Num of predicted Buildings: ", len(cont_test_1))
    print("TP, FP, FN: ")
    print(true_pos, false_pos, false_neg)
    prec, rec, f1 = measures(true_pos, false_pos, false_neg)
    print("Prec, Rec, F1: ")
    print(prec, rec, f1)

    # Test 0 real buildings but 98 predicted
    true_pos, false_pos, false_neg = IoU_masks(label3, label1)
    print("\nNum of buildings: ", len(cont_test_1))
    print("Num of predicted Buildings: ", len(cont_test_3))
    print("TP, FP, FN: ")
    print(true_pos, false_pos, false_neg)
    prec, rec, f1 = measures(true_pos, false_pos, false_neg)
    print("Prec, Rec, F1: ")
    print(prec, rec, f1)
