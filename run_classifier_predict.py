"""
  This script provides an example to wrap UER-py for classification inference.
"""
import sys
import os
import torch
import time
import argparse
import collections
import torch.nn as nn

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts
from uer.embeddings import *
from uer.encoders import *
from uer.utils.misc import pooling
from mynacos import Args

class FeatureExtractor(torch.nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)

        return output

class MyClassifier(nn.Module):
    def __init__(self, args):
        super(MyClassifier, self).__init__()
        self.labels_num = args.labels_num
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(args.emb_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.labels_num)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def extrct_feature_from_bert(args, text):
    dataset, labels = read_dataset(args, text)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    feature_vectors = []
    for i, (src_batch, seg_batch) in enumerate(batch_loader(args.batch_size, src, seg)):
        src_batch = src_batch.to(device)
        seg_batch = seg_batch.to(device)
        output = args.pre_model(src_batch, seg_batch)
        feature_vectors.append(output.cpu().detach())
    feature_vectors = torch.cat(feature_vectors, 0)

    # Vector whitening.
    # if args.whitening_size is not None:
    #     whitening = WhiteningHandle(args, feature_vectors)
    #     feature_vectors = whitening(feature_vectors, args.whitening_size, pt=True)

    print("The size of feature vectors (sentences_num * vector size): {}".format(feature_vectors.shape))
    if labels is not None:
        labels = torch.LongTensor(labels.values)
    # pp = path.split('.')[0] + '.pt'
    # torch.save({'feature': feature_vectors, 'labels': labels}, pp)
    return feature_vectors, labels

def concat_question_and_feature_vector(args, text, question_id):
    feature, labels = extrct_feature_from_bert(args, text)
    # pt_path = path.split('.')[0] + '.pt'
    # saved_tensors = torch.load(pt_path)
    # feature, labels = saved_tensors['feature'],  saved_tensors['labels']
    # labels = torch.LongTensor(labels.values)
    # 问题作为特征拼接到文本特征上
    torch.nn.functional.one_hot(torch.arange(0, 5) % 3, num_classes=5)
    return feature, labels

def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    seg_batch = None
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        if seg is not None:
            seg_batch = seg[i * batch_size: (i + 1) * batch_size]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        if seg is not None:
            seg_batch = seg[instances_num // batch_size * batch_size:]
        yield src_batch, seg_batch

def read_dataset(args, text):
    dataset, columns = [], {}
    # with open(path, mode="r", encoding="utf-8") as f:
    #     for line_id, line in enumerate(f):
    #         if line_id == 0:
    #             line = line.rstrip("\r\n").split("\t")
    #             for i, column_name in enumerate(line):
    #                 columns[column_name] = i
    #             continue
    #         line = line.rstrip("\r\n").split("\t")
    #         if "text_b" not in columns:  # Sentence classification.
    #             text_a = line[columns["text_a"]]
    #             src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
    #             seg = [1] * len(src)
    #         else:  # Sentence pair classification.
    #             text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
    #             src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
    #             src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
    #             src = src_a + src_b
    #             seg = [1] * len(src_a) + [2] * len(src_b)
    src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text) + [SEP_TOKEN])
    seg = [1] * len(src)
    if len(src) > args.seq_length:
        src = src[: args.seq_length]
        seg = seg[: args.seq_length]
    PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
    while len(src) < args.seq_length:
        src.append(PAD_ID)
        seg.append(0)
    dataset.append((src, seg))

    return dataset, None


def text_predict(r_json):
    start = int(time.time())
    print(start)
    question_id = r_json['question_id']
    text = r_json['text']
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # infer_opts(parser)
    # parser.add_argument("--labels_num", type=int, required=True,
    #                     help="Number of prediction labels.")
    # tokenizer_opts(parser)
    # parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    # parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    # parser.add_argument("--whitening_size", type=int, default=None, help="Output vector size after whitening.")
    # args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = Args()
    print(args)
    args = load_hyperparam(args)
    print(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    # Build feature extractor model.
    pre_model = FeatureExtractor(args)
    pre_model = load_model(pre_model, args.pretrained_model_path)
    pre_model.eval()
    args.pre_model = pre_model
    # Build classification model and load parameters.
    # args.soft_targets, args.soft_alpha = False, False
    model = MyClassifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    X_infer, labels = concat_question_and_feature_vector(args, text, question_id)
    batch_size = args.batch_size
    instances_num = X_infer.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()
    for i, (X, _) in enumerate(batch_loader(batch_size, X_infer, None)):
        X = X.to(device)
        with torch.no_grad():
            logits = model(X)

        pred = torch.argmax(logits, dim=1)
        pred = pred.cpu().numpy().tolist()
        prob = nn.Softmax(dim=1)(logits)
        logits = logits.cpu().numpy().tolist()
        prob = prob.cpu().numpy().tolist()
    # with open(args.prediction_path, mode="w", encoding="utf-8") as f:
    #     f.write("label")
    #     if args.output_logits:
    #         f.write("\t" + "logits")
    #     if args.output_prob:
    #         f.write("\t" + "prob")
    #     f.write("\n")
    #     for i, X in enumerate(batch_loader(batch_size, X_infer)):
    #         X = X.to(device)
    #         with torch.no_grad():
    #             logits = model(X)
    #
    #         pred = torch.argmax(logits, dim=1)
    #         pred = pred.cpu().numpy().tolist()
    #         prob = nn.Softmax(dim=1)(logits)
    #         logits = logits.cpu().numpy().tolist()
    #         prob = prob.cpu().numpy().tolist()
    #
    #         for j in range(len(pred)):
    #             f.write(str(pred[j]))
    #             if args.output_logits:
    #                 f.write("\t" + " ".join([str(v) for v in logits[j]]))
    #             if args.output_prob:
    #                 f.write("\t" + " ".join([str(v) for v in prob[j]]))
    #             f.write("\n")
    end = int(time.time())
    print(end, end-start)
    print(pred, logits, prob)
    return pred[0]

if __name__ == "__main__":
    text_predict({'model_id': '1', 'question_id': '2', 'text': '我觉得这部电影不好看'})
