# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


def entity_metrics_with_augmentation(data_index, label_index, predict, logits, label):
    pred_data = pd.DataFrame({"data_index": data_index, "label_index": label_index,
                              "pred": predict, "logit": logits, "label": label})

    new_data_index = []; new_data_logit = []; new_data_label = []; new_data_pred = []
    for target_idx in pred_data.data_index.unique():
        tgt_idx_data = pred_data[pred_data["data_index"] == target_idx]

        tgt_idx_data_0 = tgt_idx_data[tgt_idx_data["label_index"] == 0]
        tgt_idx_data_1 = tgt_idx_data[tgt_idx_data["label_index"] == 1]
        tgt_idx_data_2 = tgt_idx_data[tgt_idx_data["label_index"] == 2]
        tgt_idx_data_3 = tgt_idx_data[tgt_idx_data["label_index"] == 3]
        tgt_idx_data_4 = tgt_idx_data[tgt_idx_data["label_index"] == 4]

        pred_0 = tgt_idx_data_0.logit.iloc[0][1]
        pred_1 = tgt_idx_data_1.logit.iloc[0][1]
        pred_2 = tgt_idx_data_2.logit.iloc[0][1]
        pred_3 = tgt_idx_data_3.logit.iloc[0][1]
        pred_4 = tgt_idx_data_4.logit.iloc[0][1]

        label_0 = tgt_idx_data_0.label.iloc[0]
        label_1 = tgt_idx_data_1.label.iloc[0]
        label_2 = tgt_idx_data_2.label.iloc[0]
        label_3 = tgt_idx_data_3.label.iloc[0]
        label_4 = tgt_idx_data_4.label.iloc[0]

        new_logit = [float(pred_0), float(pred_1), float(pred_2), float(pred_3), float(pred_4)]
        new_pred = np.argmax(new_logit)
        new_label = np.argmax([label_0, label_1, label_2, label_3, label_4])

        new_data_index.append(target_idx)
        new_data_logit.append(new_logit)
        new_data_pred.append(new_pred)
        new_data_label.append(new_label)

    logit_0 = []; logit_1 = []; logit_2 = []; logit_3 = []; logit_4 = []
    label_0 = []; label_1 = []; label_2 = []; label_3 = []; label_4 = []

    for lab, logit, pred in zip(new_data_label, new_data_logit, new_data_pred):
        logit_0.append(logit[0])
        logit_1.append(logit[1])
        logit_2.append(logit[2])
        logit_3.append(logit[3])
        logit_4.append(logit[4])

        label_0.append(int(lab == 0))
        label_1.append(int(lab == 1))
        label_2.append(int(lab == 2))
        label_3.append(int(lab == 3))
        label_4.append(int(lab == 4))

    ap_0 = average_precision_score(y_true=label_0, y_score=logit_0)
    ap_1 = average_precision_score(y_true=label_1, y_score=logit_1)
    ap_2 = average_precision_score(y_true=label_2, y_score=logit_2)
    ap_3 = average_precision_score(y_true=label_3, y_score=logit_3)
    ap_4 = average_precision_score(y_true=label_4, y_score=logit_4)

    map = (ap_0 + ap_1 + ap_2 + ap_3 + ap_4) / 5
    micro_f1 = f1_score(y_true=new_data_label, y_pred=new_data_pred, average='micro')
    macro_f1 = f1_score(y_true=new_data_label, y_pred=new_data_pred, average='macro')
    f1s = f1_score(y_true=new_data_label, y_pred=new_data_pred, average=None)

    return micro_f1, macro_f1, f1s, map, [ap_0, ap_1, ap_2, ap_3, ap_4]
