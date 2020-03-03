import mxnet as mx
import numpy as np


class AccuracyWithIgnore(mx.metric.EvalMetric):
    def __init__(self, axis=1, ignore_label=None, name='Accuracy'):
        super(AccuracyWithIgnore, self).__init__(name=name)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            pred = pred.asnumpy().argmax(axis=self.axis)
            label = label.asnumpy().astype(np.int)
            self.sum_metric += np.sum((pred == label) & (label != self.ignore_label))
            self.num_inst += np.sum(label != self.ignore_label)


class CrossEntropyWithIgnore(mx.metric.EvalMetric):
    """
    CrossEntropy loss metric
    ndim: ndim of pred
    axis: axis to calculate CrossEntropy
    ignore_label: ignore_label or None
    label_weights: weights for each label
    """
    def __init__(self, ndim=2, axis=1, ignore_label=None, label_weights=None, eps=1e-12, name='CrossEntropy'):
        super(CrossEntropyWithIgnore, self).__init__(name=name)
        self.axis = axis
        self.ignore_label = ignore_label
        self.label_weights = label_weights
        if isinstance(self.label_weights, list):
            self.label_weights = np.array(self.label_weights)
        self.eps = eps
        self.axes = list(range(ndim))
        self.axes.remove(axis)
        self.axes.insert(ndim-1, axis)

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            pred = pred.asnumpy()
            label = label.asnumpy().astype(np.int).ravel()
            pred = pred.transpose(self.axes).reshape((-1, pred.shape[self.axis]))
            prob = pred[np.arange(label.size), label]
            loss = -np.log(prob+self.eps)
            if self.label_weights is None:
                weight = np.ones(label.size)
            else:
                weight = self.label_weights[label]
            if self.ignore_label is None:
                mask = np.ones(label.size)
            else:
                mask = (label != self.ignore_label)
            self.sum_metric += float(np.sum(loss * mask))
            self.num_inst += int(np.sum(mask))


class PerClassMetric(mx.metric.EvalMetric):
    """
    Base class for per-class metric
    num_class: number of classes
    class_dict: int -> str, dict of class_id -> class_name
    round_digit: round to how many digits
    report_occurrence: whether to report the #occurrences of each class
    """
    def __init__(self, num_class, class_dict={}, round_digit=6, report_occurrence=False, name='PerClassMetric'):
        self.num_class = num_class
        super(PerClassMetric, self).__init__(name=name)
        self.class_dict = class_dict
        self.round_digit = round_digit
        self.report_occurrence = report_occurrence
        self.reset()

    def reset(self):
        self.num_inst = np.zeros(self.num_class)
        self.sum_metric = np.zeros(self.num_class)

    def update(self, labels, preds):
        raise NotImplementedError

    def get(self):
        res = []
        for j in range(self.num_class):
            if j != self.ignore_label:
                class_name = self.class_dict.get(j, str(j))
                if self.num_inst[j] == 0:
                    value = float('nan')
                else:
                    value = np.round(self.sum_metric[j] / self.num_inst[j], self.round_digit)
                if self.report_occurrence:
                    res.append((class_name, float(value), int(self.num_inst[j])))
                else:
                    res.append((class_name, float(value)))
        return self.name, res


class PerClassAccuracy(PerClassMetric):
    """
    Per class accuracy
    num_class: number of classes
    axis: axis of softmax for argmax
    ignore_label: label to ignore
    class_dict: int -> str, dict of class_id -> class_name
    round_digit: round to how many digits
    report_occurrence: whether to report the #occurrences of each class
    """
    def __init__(self, num_class, axis=1, ignore_label=None, class_dict={}, round_digit=6, report_occurrence=False, name='PerClassAccuracy'):
        super(PerClassAccuracy, self).__init__(num_class, class_dict, round_digit, report_occurrence, name)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            pred = pred.asnumpy().argmax(axis=self.axis)
            # print("pred",pred)
            label = label.asnumpy().astype(np.int)
            # print("label",label)
            for j in range(self.num_class):
                if j != self.ignore_label:
                    self.sum_metric[j] += np.sum((pred == label) & (label == j))
                    self.num_inst[j] += np.sum(label == j)

class PerClassIoU(PerClassMetric):
    """
    Per class IoU
    num_class: number of classes
    axis: axis of softmax for argmax
    ignore_label: label to ignore
    class_dict: int -> str, dict of class_id -> class_name
    round_digit: round to how many digits
    report_occurrence: whether to report the #occurrences of each class
    """
    def __init__(self, num_class, axis=1, ignore_label=None, class_dict={}, round_digit=6, report_occurrence=False, name='PerClassIoU'):
        super(PerClassIoU, self).__init__(num_class, class_dict, round_digit, report_occurrence, name)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            pred = pred.asnumpy().argmax(axis=self.axis)
            label = label.asnumpy().astype(np.int)
            for j in range(self.num_class):
                if j != self.ignore_label:
                    self.sum_metric[j] += np.sum((pred == j) & (label == j))
                    self.num_inst[j] += np.sum(((pred == j) & (label != self.ignore_label)) | (label == j))


class ConfusionMatrix(mx.metric.EvalMetric):
    """
    report the confusion matrix
    self.matrix[i,j]: gt_label=i, pred=j
    """
    def __init__(self, num_class, axis=1, ignore_label=None, name='ConfusionMatrix'):
        self.num_class = num_class
        super(ConfusionMatrix, self).__init__(name=name)
        self.axis = axis
        self.ignore_label = ignore_label
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_class, self.num_class), dtype=np.int)

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            pred = pred.asnumpy().argmax(axis=self.axis).ravel()
            label = label.asnumpy().astype(np.int).ravel()
            for j in range(label.size):
                if label[j] != self.ignore_label:
                    self.matrix[label[j], pred[j]] += 1

    def get(self):
        return self.name, self.matrix



