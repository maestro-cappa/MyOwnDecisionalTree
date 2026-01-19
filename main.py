from ML import ML
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# My own ML DecisionalTree algorithim that can run some sklearn datasets (digits, breast_cancer, wine...) with performance given (errors and accuracy)


#loading dataset
data = load_digits()

# getting and splitting datas
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # method to get datas from dataset (80% train data, 20% test data)


#formatting datas in [feature1, feature2..., featureN, label] format so the cusotom model can read it
X_train = X_train.tolist()
y_train = y_train.tolist()
X_test = X_test.tolist()
y_test = y_test.tolist()
dataset = [list(features) + [label] for features, label in zip(X_train, y_train)]
tests = [list(features) + [label] for features, label in zip(X_test, y_test)]


ml = ML(dataset)

ml.train(ml.root)
for sample in tests:
    ml.guess(ml.root, sample)


def check_score(tests, score):
    errors = sum(el1 != el2 for el1, el2 in zip(tests, score))
    tot_tests = len(y_test)
    accuracy = (tot_tests - errors) / tot_tests * 100
    approximate_accuracy = round(accuracy, 3)
    print(f"ERRORS: {errors}")
    print(f"ACCURACY: {approximate_accuracy}%")


print(y_test)
check_score(y_test, ml.score)