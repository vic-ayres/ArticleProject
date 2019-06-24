import numpy as np

class ConfusionMatrix(object):

    def __init__(self, labels, actual, predicted):
        self.labels = labels
        self.actual = actual
        self.predicted = predicted

    def count_actual(self):
        num = []
        for label in self.labels:
            count = 0
            for value in self.actual:
                if(label == value):
                    count+=1
            num.append(count)
        return np.array(num)


    def count_predicted(self):
        num = []
        for label in self.labels:
            count = 0
            for value in self.predicted:
                if(label == value):
                    count+=1
            num.append(count)
        return np.array(num)

    def count_combination(self,label1, label2):
        count = 0
        for i in range(0, len(self.actual)):
            if (self.actual[i] == label1 and self.predicted[i] == label2):
                    count +=1
        return count

    def create_matrix(self):
        cm = []
        for i in range(0, len(self.labels)):
            row = []
            for j in range(0, len(self.labels)):
                count = self.count_combination(self.labels[i], self.labels[j])
                row.append(count)
            cm.append(row)
        return np.array(cm)

    def max_label(self):
        max = self.labels[0]
        for i in range(1,len(self.labels)):
            if(len(max)<len(self.labels[i])):
                max = self.labels[i]
        return max
            

    def print_matrix(self):
        cm = self.create_matrix()
        max = len(self.max_label())+3
        format_string1 = "{:>"+str(max)+"}"
        format_string2 = "{:"+str(max)+"}"
        format_string3 = "%"+str(max)+"d"
        print(" "*max, end = " ")
        for i in range(0,len(self.labels)):
            print(format_string1.format(self.labels[i]+"(P)"), end = " ")
        print()
        for i in range(0,len(self.labels)):
            print(format_string2.format(self.labels[i]+"(A)"), end = " ")
            for j in range(0,len(self.labels)):
                print(format_string3 % cm[i,j], end = " " )
            print()
       




    
