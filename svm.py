
'''This is classifier for MNIST dataset. First we have to import our mnist_loader 
which takes care of loading and preprocessing our data. We import 
support vector machine(svm) from sklearn. i really like this approach to the problem,
because it is simple and uses not so popular svm.'''


import loader 
from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = loader.load_data()
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))

if __name__ == "__main__":
    svm_baseline()
    