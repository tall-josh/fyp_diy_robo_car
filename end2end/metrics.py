import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

""" TO DO: Fit true and predicted throttle """
class Metrics():

    def __init__(self, classes):
        '''
        classes   = list of classes ie: ["dog", "cat", "rat"]
                                    or: [0, 1, 2]
        '''
        self.classes     = classes
        self.num_classes = len(classes)
        # One-hot embedded labels
        self.oh_true_steering = []
        # Probability distributions as predicted by network
        self.probs  = []
        # Expected value from the prob distribution
        self.expected_bins = []
        # Actual steering values (0 - 1023)
        # ???
        self.throttle = []

    def update(self, probs, expected_bins, throttle, annos):
        self.oh_true_steering.extend(label_binarize(annos["steering"], classes=self.classes))
        self.probs.extend(probs)
        self.expected_bins.extend(expected_bins)
        self.throttle.extend(annos["throttle"])

    def reset(self):
        self.oh_true_steering = []
        self.probs    = []
        self.expected_bins = []
        self.throttle = []

    def compute_metrics(self, save_dir=None):

        self.oh_true_steering = np.array(self.oh_true_steering)
        self.probs = np.array(self.probs)

        precision = dict()
        recall = dict()
        thresh = dict()
        for c in range(self.num_classes):

            # Precision Recall
            precision[c], recall[c], thresh[c] = precision_recall_curve(self.oh_true_steering[:, c], self.probs[:, c])

        # Average Precision per class
        average_precision = {c:ap for c,ap in enumerate(average_precision_score(self.oh_true_steering, self.probs, None))}

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], thresh["micro"] = precision_recall_curve(self.oh_true_steering.ravel(),
            self.probs.ravel())

        # MAP...I think
        average_precision["micro"] = average_precision_score(self.oh_true_steering, self.probs,
                                                             average="micro")

        metrics = {}
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["thresh"] = thresh
        metrics["average_precision"] = average_precision
        metrics["predictions"] = self.expected_bins
        metrics["true_steering"] = [np.argmax(x) for x in self.oh_true_steering]
        metrics["true_throttle"] = self.throttle
        if save_dir is not None:
            SaveEvalFigs(metrics, save_dir)

        return metrics


    def SaveEvalFigs(self, metrics, save_dir):
        print(f"save_dir: {save_dir}")
        print(f"metrics: {metrics.keys()}")

        precision = metrics["precision"]
        recall    = metrics["recall"]
        ave_prec  = metrics["average_precision"]

        # Precision Recall curves for each class
        for c in range(len(self.classes)):

            fig = plt.figure(figsize=(8,6))
            plt.step(recall[c], precision[c], color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall[c], precision[c], step='post', alpha=0.2,
                             color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            ap = ave_prec[c]
            plt.title(f'Class: {c}, AP: {ap:0.2f}')
            path = os.path.join(save_dir, f"precision_recall_class_{c:02d}.jpg")
            plt.savefig(path)

        # Precision Recall curve over all classes
        mean_ap = ave_prec["micro"]
        fig = plt.figure(figsize=(8,6))
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Average precision score, micro-averaged over all classes: MAP: {mean_ap:0.2f}')
        path = os.path.join(save_dir, f"precision_recall_micro_average.jpg")
        plt.savefig(path)

        # Predicted steering vs Label scatter
        preds = metrics["predictions"]
        annos = metrics["annotations"]
        fig = plt.figure(figsize=(8,6))
        plt.plot(annos, preds, 'b.')
        plt.title(f'Annotation vs Prediction')
        plt.xlabel("annotations")
        plt.ylabel("predictions")
        path = os.path.join(save_dir, f"prediction_vs_annotation.jpg")
        fig.savefig(path)

        # Predicted throttle vs Label scatter
        preds = metrics["predictions"]
        annos = metrics["annotations"]
        fig = plt.figure(figsize=(8,6))
        plt.plot(annos, preds, 'b.')
        plt.title(f'Annotation vs Prediction')
        plt.xlabel("annotations")
        plt.ylabel("predictions")
        path = os.path.join(save_dir, f"prediction_vs_annotation.jpg")
        fig.savefig(path)




