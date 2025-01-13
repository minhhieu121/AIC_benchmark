import numpy as np

class Evaluator:
    def __init__(self, label, result):
        self.label = label
        self.result = result
        self.r1 = np.zeros(len(label), dtype=bool)
        self.r5 = np.zeros(len(label), dtype=bool)
        self.r10 = np.zeros(len(label), dtype=bool)

    def compute_result(self):
        num_sample = len(self.label)
        for i in range(num_sample):
            current = self.result[i]
            for j in range(len(current)):
                print(f"Resutl: {current[j]} - Label: {self.label[i]}")
                if (current[j] == self.label[i]):
                    if j == 0:
                        self.r1[i] = True
                        self.r5[i] =  True
                        self.r10[i] = True
                    elif j < 5:
                        self.r5[i] = True
                        self.r10[i] = True
                    elif j < 10:
                        self.r10[i] = True
                    break

    def perform_evaluation(self):
        self.compute_result()
        recall1 = np.mean(self.r1)
        recall5 = np.mean(self.r5)
        recall10 = np.mean(self.r10)
        
        return recall1, recall5, recall10

