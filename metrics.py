import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

class Metrics():
    
    def __init__(self, classes, ):
        '''
        classes   = list of classes ie: ["dog", "cat", "rat"]
                                    or: [0, 1, 2]
        threholds = [0.1, 0.2, 0.3, ...] 
        '''
        self.classes     = classes
        self.num_classes = len(classes)
        # One-hot embedded labels
        self.oh_annos = []
        # Probability distributions as predicted by network
        self.probs  = []
        # Expected value from the prob distribution
        self.expected_bins = []
        # Actual steering values (0 - 1023)
                        
    def update(self, probs, expected_bins, annos):
        assert len(probs) == len(annos), "Number of predictions and labels must match."
        
        self.oh_annos.extend(label_binarize(annos, classes=self.classes))
        self.probs.extend(probs)
        self.expected_bins.extend(expected_bins)
        
    def reset(self):
        self.oh_annos = []
        self.probs    = []
        self.expected_bins = []
                        
    def compute_metrics(self, save_dir=None):
        
        # This is bloody jank, but I can't think of a better way atm
        self.oh_annos = np.array(self.oh_annos)
        self.probs = np.array(self.probs)
        
        precision = dict()
        recall = dict()
        thresh = dict()
        for c in range(self.num_classes):
            
            # Precision Recall
            precision[c], recall[c], thresh[c] = precision_recall_curve(self.oh_annos[:, c], self.probs[:, c])
        
        # Average Precision per class
        average_precision = {c:ap for c,ap in enumerate(average_precision_score(self.oh_annos, self.probs, None))}
        
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], thresh["micro"] = precision_recall_curve(self.oh_annos.ravel(),
            self.probs.ravel())
        
        # MAP...I think
        average_precision["micro"] = average_precision_score(self.oh_annos, self.probs,
                                                             average="micro")
        
        result = {}
        result["precision"] = precision
        result["recall"] = recall
        result["thresh"] = thresh
        result["average_precision"] = average_precision
        result["predictions"] = self.expected_bins
        result["annotations"] = [np.argmax(x) for x in self.oh_annos]
        
        if save_dir is not None:
            SaveEvalFigs(result, save_dir)
        
        return result

    
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
        
        # Predicted vs Label scatter
        preds = metrics["predictions"]
        annos = metrics["annotations"]
        fig = plt.figure(figsize=(8,6))
        plt.plot(annos, preds, 'b.')
        plt.title(f'Annotation vs Prediction')
        plt.xlabel("annotations")
        plt.ylabel("predictions")
        path = os.path.join(save_dir, f"prediction_vs_annotation.jpg")
        fig.savefig(path)
        
        
        
        