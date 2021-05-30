# -*- encoding: utf-8 -*-
import os, logging, random, time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from metrics import *


class TransformerClassifierWithAuxiliarySentence(nn.Module):
    def __init__(self, encoder, temperature=1.0):
        super(TransformerClassifierWithAuxiliarySentence, self).__init__()
        self.cls_num = 2
        self.encoder = encoder
        self.out = nn.Linear(768, self.cls_num)
        self.temperature = temperature

    def forward(self, input, attention_mask):
        _, out = self.encoder(input, attention_mask=attention_mask)
        out = self.out(out)
        out = out / self.temperature

        return out

    def __str__(self):
        return 'TransformerClassifierWithAuxiliarySentence'


class NewsDataset(Dataset):
    def __init__(self, data_index, label_index, input_id, attention_mask, y):
        self.data_index = data_index
        self.label_index = label_index
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.y = y

    def __len__(self):
        return len(self.input_id)

    def __getitem__(self, idx):
        return self.data_index[idx], self.label_index[idx], \
               self.input_id[idx], self.attention_mask[idx], self.y[idx]


def proces_line(sent, label, pretrain_type, input_type):
    entstart_first_index = sent.find("[")
    entend_first_index = sent.find("]")
    ent1_token = sent[entstart_first_index:(entend_first_index + 1)]

    entstart_second_index = sent.find("[", entstart_first_index + 1)
    entend_second_index = sent.find("]", entend_first_index + 1)
    ent2_token = sent[entstart_second_index:(entend_second_index + 1)]

    sent = sent.replace(ent1_token, "[Ent1]")
    sent = sent.replace(ent2_token, "[Ent2]")
    sent = sent.strip('\n')
    line = sent.strip(' ')

    if input_type == "T":
        aux_questions = ["Do [Ent1] and [Ent2] have neutral sentiment toward each other?",
                         "Does [Ent1] has positive sentiment toward [Ent2]?",
                         "Does [Ent2] has positive sentiment toward [Ent1]?",
                         "Does [Ent1] has negative sentiment toward [Ent2]?",
                         "Does [Ent2] has negative sentiment toward [Ent1]?"]
    elif input_type == "P":
        aux_questions = ["[Ent1] - [Ent2] - neutral",
                         "[Ent1] - [Ent2] - positive",
                         "[Ent2] - [Ent1] - positive",
                         "[Ent1] - [Ent2] - negative",
                         "[Ent2] - [Ent1] - negative"]
    else:
        raise TypeError

    combined_sentences = []
    for question in aux_questions:
        if pretrain_type == "roberta":
            sent = f"<s> {line} </s></s> {question} </s>"
        elif pretrain_type == "spanbert":
            sent = f"[CLS] {line} [SEP] {question} [SEP]"
        else:
            raise TypeError

        combined_sentences.append(sent)

    labels = np.zeros(5, dtype=int)
    labels[int(label)] += 1
    labels = labels.tolist()

    return combined_sentences, labels


def get_loader(data, tokenizer, max_length, bsz=32):
    data_index = [i[0] for i in data]
    label_index = [i[1] for i in data]
    input = [i[2] for i in data]
    label = [i[3] for i in data]
    input = tokenizer.batch_encode_plus(input, add_special_tokens=False, max_length=max_length,
                                        pad_to_max_length=True, return_tensors='pt')
    input_ids = input['input_ids']
    attention_masks = input['attention_mask']

    dataset = NewsDataset(data_index, label_index, input_ids, attention_masks, label)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False)
    return dataloader


def main(args, train_path, logger, exp_id):
    train = []
    valid = []
    test = []

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    train_df.columns = ["Sent", "Ent1", "Ent2", "Label"]
    for idx, row in train_df.iterrows():
        inputs, labels = proces_line(row["Sent"], row["Label"], args.pretrain_type, args.input_type)
        for label_idx, (input, label) in enumerate(zip(inputs, labels)):
            train.append((idx, label_idx, input, label))
    del train_df

    valid_df = pd.read_csv("dataset/valid_reduced.txt", sep="\t", header=None)
    valid_df.columns = ["Sent", "Ent1", "Ent2", "Label"]
    for idx, row in valid_df.iterrows():
        inputs, labels = proces_line(row["Sent"], row["Label"], args.pretrain_type, args.input_type)
        for label_idx, (input, label) in enumerate(zip(inputs, labels)):
            valid.append((idx, label_idx, input, label))
    del valid_df

    test_df = pd.read_csv("dataset/test_reduced.txt", sep="\t", header=None)
    test_df.columns = ["Sent", "Ent1", "Ent2", "Label"]
    for idx, row in test_df.iterrows():
        inputs, labels = proces_line(row["Sent"], row["Label"], args.pretrain_type, args.input_type)
        for label_idx, (input, label) in enumerate(zip(inputs, labels)):
            test.append((idx, label_idx, input, label))
    del test_df

    target_iter = len(train_df.index) * 5 * args.max_epoch # * 5 (augmented data)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    encoder = RobertaModel.from_pretrained('roberta-base')


    model = TransformerClassifierWithAuxiliarySentence(encoder, args.temperature)
    train_loader = get_loader(train, tokenizer, max_length=125, bsz=args.batch_size)
    test_loader = get_loader(test, tokenizer, max_length=125, bsz=args.batch_size)
    val_loader = get_loader(valid, tokenizer, max_length=125, bsz=args.batch_size)

    t_total = (target_iter // args.batch_size) + 1
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1, #args.weight_decay
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-06)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    model = model.cuda()
    loss_ce = nn.CrossEntropyLoss(reduction='none')
    performance_logs = {"valid": [], "test": []}
    logit_logs = []; label_logs = []
    time_diffs = []

    for epoch in range(args.max_epoch):
        model.train()
        train_loss = []
        data_index_list = []
        label_index_list = []
        label_list = []
        logit_list = []
        pred_list = []

        start = time.time()

        for idx, batch in enumerate(train_loader):
            model.train()
            data_index, label_index, input_id, attention_mask, y = batch
            input_id, attention_mask, y = input_id.cuda(), attention_mask.cuda(), y.cuda()
            pred = model(input_id, attention_mask)
            loss_all = loss_ce(pred, y)
            loss = torch.mean(loss_all)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            train_loss.append(loss.item())
            label_list += y.tolist()

            data_index_list.extend(data_index)
            label_index_list.extend(label_index)

            logit_list += F.softmax(pred, dim=-1).tolist()
            pred_list += pred.argmax(1).tolist()

            if idx % 10 == 0:
                logger.info(f"Epoch #{epoch+1} Batch #{idx} Loss {loss:.4f}")
        end = time.time()
        time_diffs.append(end-start)

        print_loss = np.mean(train_loss)
        logger.info(f'epoch: {epoch+1} | train loss: {print_loss:.4f}')

        for datatype, loader in ([['valid', val_loader], ['test', test_loader]]):
            total_loss = []
            data_index_list = []
            label_index_list = []
            label_list = []
            logit_list = []
            pred_list = []

            model.eval()
            for batch in loader:
                with torch.no_grad():
                    data_index, label_index, input_id, attention_mask, y = batch
                    input_id, attention_mask, y = input_id.cuda(), attention_mask.cuda(), y.cuda()
                    pred = model(input_id, attention_mask)
                    loss_all = loss_ce(pred, y)
                    loss = torch.mean(loss_all)

                    total_loss.append(loss.item())
                    label_list += y.tolist()

                    data_index_list.extend(data_index)
                    label_index_list.extend(label_index)

                    logit_list += F.softmax(pred, dim=-1).tolist()
                    pred_list += pred.argmax(1).tolist()

            print_loss = np.mean(total_loss)
            accuracy, macro_f1, f1_cls, map, ap_cls = \
                entity_metrics_with_augmentation(data_index_list, label_index_list, pred_list, logit_list, label_list)
            log = [print_loss, accuracy, macro_f1, f1_cls[0], f1_cls[1], f1_cls[2], f1_cls[3], f1_cls[4],
                   map, ap_cls[0], ap_cls[1], ap_cls[2], ap_cls[3], ap_cls[4]]

            performance_logs[datatype].append(log)
            if datatype == "test":
                logit_logs.append(logit_list)
                label_logs.append(label_list)

            logger.info(f'epoch: {epoch+1} | {datatype} loss: {print_loss:.4f}  acc: {accuracy:.4f}  '
                        f'f1: {macro_f1:.4f}  f1-0: {f1_cls[0]:.4f}  f1-1: {f1_cls[1]:.4f}  f1-2: {f1_cls[2]:.4f}  '
                        f'f1-3: {f1_cls[0]:.4f}  f1-4: {f1_cls[1]:.4f}  '
                        f'map: {map:.4f}  ap0: {ap_cls[0]:.4f}  ap1: {ap_cls[1]:.4f}  ap2: {ap_cls[2]:.4f}  '
                        f'ap3: {ap_cls[3]:.4f}  ap4: {ap_cls[4]:.4f}')

    logger.info(f"epoch time : {np.mean(time_diffs)}")

    best_epoch = None
    best_loss = np.inf
    for idx, valid_log in enumerate(performance_logs["valid"]):
        epoch_loss = valid_log[0]  # loss, accuracy, macro_f1, ...
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = idx

    valid_performance = performance_logs["valid"][best_epoch]
    test_performance = performance_logs["test"][best_epoch]
    test_logit = logit_logs[best_epoch]
    test_label = label_logs[best_epoch]

    if not os.path.exists("out"):
        os.makedirs("out")
    torch.save(model.state_dict(), f"out/model_{exp_id}.pt")

    df = pd.DataFrame({"logit": test_logit, "label": test_label})
    df.to_csv(f"out/logit_{exp_id}.csv", index=False)

    col = ['loss', 'acc', 'macro-f1', 'f1-0', 'f1-1', 'f1-2', 'f1-3', 'f1-4', 'map', 'ap0', 'ap1', 'ap2', 'ap3', 'ap4']
    df = pd.DataFrame([valid_performance], columns=col)
    df.to_csv(f'out/perf_{exp_id}_valid.csv', index=False)
    print("====== PERFORMANCE ON VALID SET ======")
    print(f"MICRO-F1: {df['micro-f1']} | MACRO-F1: {df['macro-f1']} | mAP: {df['map']}")

    df = pd.DataFrame([test_performance], columns=col)
    df.to_csv(f'out/perf_{exp_id}_test.csv', index=False)
    print("====== PERFORMANCE ON TEST SET ======")
    print(f"MICRO-F1: {df['micro-f1']} | MACRO-F1: {df['macro-f1']} | mAP: {df['map']}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", type=str, choices=["T", "P"], required=True)
    parser.add_argument("--resample", type=str, choices=["none", "up", "down"], required=True)
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--random_seed", type=int, default=20180422)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.resample == "none":
        train_path = "dataset/train.txt"
    elif args.resample == "up":
        train_path = "dataset/train_over.txt"
    elif args.resample == "down":
        train_path = "dataset/train_under.txt"
    else:
        raise TypeError

    exp_id = f"DSE2QA_{args.input_type}_{args.resample}_{args.max_epoch}_{args.batch_size}_{args.random_seed}"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(exp_id)
    f_handler = logging.FileHandler(f"out/{exp_id}.txt", mode="w")
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)

    main(args, train_path, logger, exp_id)
