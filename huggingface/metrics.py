from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_compute_metrics(level):
    assert level in [1, 3]

    def compute_metrics_level1(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def compute_metrics_level3(eval_pred):
        predictions, labels = eval_pred

        mask = predictions[0].argmax(axis=-1)
        gender = predictions[1].argmax(axis=-1)
        age = predictions[2].argmax(axis=-1)

        preds = mask * 6 + gender * 3 + age
        labels = labels[:, 0] * 6 + labels[:, 1] * 3 + labels[:, 2]

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    compute_metrics = compute_metrics_level1 if level == 1 else compute_metrics_level3
    return compute_metrics
