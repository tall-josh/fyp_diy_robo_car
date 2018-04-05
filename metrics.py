import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

class Metrics():
    
    def __init__(self, classes):
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
        
                        
    def update(self, probs, annos):
        assert len(probs) == len(annos), "Number of predictions and labels must match."
        
        self.oh_annos.extend(label_binarize(annos, classes=self.classes))
        self.probs.extend(probs)
        
    def reset(self):
        self.oh_annos = []
        self.probs    = []
                        
    def compute_metrics(self):
        
        # This is bloody jank, but I can't think of a better way atm
        self.oh_annos = np.array(self.oh_annos)
        self.probs = np.array(self.probs)
        
        precision = dict()
        recall = dict()
        thresh = dict()
        average_precision = dict()
        for c in range(self.num_classes):
            precision[c], recall[c], thresh[c] = precision_recall_curve(self.oh_annos[:, c], self.probs[:, c])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], thresh["micro"] = precision_recall_curve(self.oh_annos.ravel(),
            self.probs.ravel())
        
        average_precision["micro"] = average_precision_score(self.oh_annos, self.probs,
                                                             average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(average_precision["micro"]))
        return precision, recall, thresh, average_precision
    
