import csv
import os
from os.path import join

import imageio
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from PIL import Image
from scipy.ndimage import zoom
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, num_classes=16):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,torch.Tensor):
            assert len(alpha)==num_classes
            self.alpha = alpha
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0].fill_(alpha)
            self.alpha[1:].fill_(1-alpha)
    def forward(self, input, target):
        temp_input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.num_classes)
        temp_target = target.view(-1, 1)
        logpt = F.log_softmax(temp_input, dim=1)
        logpt = logpt.gather(1, temp_target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def output_up(output, target):
        ph, pw = output.size(2), output.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            output = F.interpolate(input=output, size=(h, w), mode='bilinear', align_corners=True)
        return output

def lr_schedule_cosine(lr_min, lr_max, per_epochs):
    def compute(epoch):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / per_epochs * np.pi))
    return compute

def _fast_hist(label_true, label_pred, n_class):

    mask = (label_true >= 0) & (label_true < n_class)
    hist = numpy.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist.astype(int)

def acc_etc(label_trues, label_preds, n_class, test=False):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt, lp, n_class)
    diagonal = np.diag(hist)
    pixel_acc = diagonal.sum() / np.maximum(np.sum(hist), 1)

    Recall = diagonal / np.maximum(hist.sum(1), 1)
    Recall = np.where(np.isnan(Recall), 0, Recall)
    M_Recall = np.nanmean(Recall)

    Pre = diagonal / np.maximum(hist.sum(0), 1)
    Pre = np.where(np.isnan(Pre), 0, Pre)
    M_Pre = np.nanmean(Pre)

    F1_score = 2*(Pre*Recall)/(Pre+Recall)
    M_F1_score = 2*(M_Pre*M_Recall)/(M_Pre+M_Recall)
    # po = acc
    # pe = np.dot(hist.sum(axis=1), hist.sum(axis=0)) / float(hist.sum() ** 2)
    # Kappa = (po - pe) / (1 - pe)

    iou = diagonal / (hist.sum(axis=1) + hist.sum(axis=0) - diagonal)
    MIOU = np.nanmean(iou)
    MIOU_no_back = np.nanmean(iou[1:])

    if test:
        return pixel_acc, M_F1_score, MIOU, hist, iou, F1_score, Recall, Pre
    else:
        return pixel_acc, iou, M_Pre, M_Recall, M_F1_score, MIOU_no_back, MIOU

def _fast_hist_combine(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class) & (label_pred >= 0) & (label_pred < n_class)
    hist = numpy.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def acc_etc_combine(label_trues, label_preds, n_class, class_mapping=None):
    if class_mapping is None:
        class_mapping = {i: i for i in range(n_class)}
    hist = numpy.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        lt_mapped = numpy.vectorize(class_mapping.get)(lt)
        lp_mapped = numpy.vectorize(class_mapping.get)(lp)
        hist += _fast_hist_combine(lt_mapped, lp_mapped, n_class)

    acc = numpy.diag(hist).sum() / numpy.maximum(numpy.sum(hist), 1)

    Recall = numpy.diag(hist) / numpy.maximum(hist.sum(1), 1)
    M_Recall = numpy.nanmean(Recall)

    Pre = numpy.diag(hist) / numpy.maximum(hist.sum(0), 1)
    M_Pre = numpy.nanmean(Pre)

    F1_score = 2 * (Pre * Recall) / (Pre + Recall)
    M_F1_score = 2 * (M_Pre * M_Recall) / (M_Pre + M_Recall)

    po = acc
    pe = numpy.dot(hist.sum(axis=1), hist.sum(axis=0)) / float(hist.sum() ** 2)
    Kappa = (po - pe) / (1 - pe)

    iou = numpy.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - numpy.diag(hist))
    MIOU = numpy.nanmean(iou)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

    return hist, iou, F1_score, Recall, Pre


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['masks']
    # low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def draw_ROC(output,img_label,out_path):  # output.shape(B,C,H,W),  label.shape(B,H,W)
        num_classes = num_classes
        y_scores = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, num_classes).cpu()
        y_true = img_label.view(-1, 1).cpu()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve((y_true == i).int(), y_scores[:, i])  # 这对吗？对一类对应通道0？应该不对
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('{}/ROC.png'.format(out_path))
        # plt.show()


def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.4f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(hist, IoUs, F1_Score, Recall, Precision, miou_out_path, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(numpy.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(F1_Score, name_classes, "mF1_Score = {0:.2f}%".format(numpy.nanmean(F1_Score)*100), "F1_Score", \
        os.path.join(miou_out_path, "F1_Score.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save F1_Score out to " + os.path.join(miou_out_path, "F1_Score.png"))

    draw_plot_func(Recall, name_classes, "mRecall = {0:.2f}%".format(numpy.nanmean(Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(numpy.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    # with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
    #     writer          = csv.writer(f)
    #     writer_list     = []
    #     writer_list.append([' '] + [str(c) for c in name_classes])
    #     for i in range(len(hist)):
    #         writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
    #     writer.writerows(writer_list)
    # print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        actual_classes = []
        for i in range(len(hist)):
            if sum(hist[i]) > 0:
                actual_classes.append(name_classes[i])

        class_list = [' '] + [str(c) for c in actual_classes]
        writer_list.append(class_list)
        for i in range(len(hist)):
            if sum(hist[i]) > 0:
                writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))

    csv_file_path = os.path.join(miou_out_path, "confusion_matrix.csv")
    confusion_matrix_data = pd.read_csv(csv_file_path, index_col=0)

    normalized_confusion_matrix = confusion_matrix_data / confusion_matrix_data.values.max() * 100
    plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.2)
    heatmap = sns.heatmap(normalized_confusion_matrix, annot=True, cmap='RdPu', fmt='.1f', cbar=True,
                        annot_kws={"weight": "bold"}, cbar_kws={"label": "Scale", "orientation": "vertical"})

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', weight='bold')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right', weight='bold')
    plt.tight_layout()
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')
    plt.title('Normalized Confusion Matrix (0-100)')
    plt.savefig(os.path.join(miou_out_path, 'confusion_matrix_heatmap.png'))
    print("Save confusion_matrix heatmap out to " + os.path.join(miou_out_path, "confusion_matrix_heatmap.png"))


