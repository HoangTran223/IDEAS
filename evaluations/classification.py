import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import logging


def evaluate_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale', tune=False):
    if tune:
        results = {
            'acc': 0,
            'macro-F1': 0
        }
        logger = logging.getLogger('main')
        for C in [0.1, 1, 10, 100, 1000]:
            for gamma in ['scale', 'auto', 10, 1, 0.1, 0.01, 0.001]:
                print(f'C: {C}, gamma: {gamma}')
                for kernel in ['rbf', 'linear']:
                    logger.info(f'C: {C}, gamma: {gamma}, kernel: {kernel}')
                    if classifier == 'SVM':
                        clf = SVC(C=C, kernel=kernel, gamma=gamma)
                    else:
                        raise NotImplementedError

                    clf.fit(train_theta, train_labels)
                    preds = clf.predict(test_theta)
                    this_results = {
                        'acc': accuracy_score(test_labels, preds),
                        'macro-F1': f1_score(test_labels, preds, average='macro')
                    }
                    results = {
                        key: max(results[key], this_results[key])
                        for key in results
                    }
                    logger.info(f'Accuracy: {this_results["acc"]}, Macro-F1: {this_results["macro-F1"]}')
    else:
        if classifier == 'SVM':
            clf = SVC(gamma=gamma)
        else:
            raise NotImplementedError

        clf.fit(train_theta, train_labels)
        preds = clf.predict(test_theta)
        results = {
            'acc': accuracy_score(test_labels, preds),
            'macro-F1': f1_score(test_labels, preds, average='macro')
        }
    return results


def crosslingual_classification(
    train_theta_en,
    train_theta_cn,
    test_theta_en,
    test_theta_cn,
    train_labels_en,
    train_labels_cn,
    test_labels_en,
    test_labels_cn,
    classifier="SVM",
    gamma="scale"
):
    intra_en = evaluate_classification(train_theta_en, test_theta_en, train_labels_en, test_labels_en, classifier, gamma)
    intra_cn = evaluate_classification(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn, classifier, gamma)

    cross_en = evaluate_classification(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en, classifier, gamma)
    cross_cn = evaluate_classification(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn, classifier, gamma)

    return {
        'intra_en': intra_en,
        'intra_cn': intra_cn,
        'cross_en': cross_en,
        'cross_cn': cross_cn
    }


def hierarchical_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale'):
    num_layer = len(train_theta)
    results = defaultdict(list)

    for layer in range(num_layer):
        layer_results = evaluate_classification(train_theta[layer], test_theta[layer], train_labels, test_labels, classifier, gamma)

        for key in layer_results:
            results[key].append(layer_results[key])

    for key in results:
        results[key] = np.mean(results[key])

    return results
