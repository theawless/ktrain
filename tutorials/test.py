import sklearn.datasets

import ktrain
import ktrain.text


def main():
    categories = ['comp.graphics', 'soc.religion.christian']
    train_b = sklearn.datasets.fetch_20newsgroups(subset='train', categories=categories,
                                                  shuffle=True, random_state=42)
    test_b = sklearn.datasets.fetch_20newsgroups(subset='test', categories=categories,
                                                 shuffle=True, random_state=42)
    x_train = train_b.data
    y_train = train_b.target
    x_test = test_b.data
    y_test = test_b.target

    transformer = ktrain.text.Transformer('distilbert-base-uncased', maxlen=64,
                                          class_names=train_b.target_names)
    trn = transformer.preprocess_train(x_train, y_train)
    val = transformer.preprocess_test(x_test, y_test)
    model = transformer.get_classifier()

    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=8)
    learner.fit_onecycle(8e-5, 2)


if __name__ == "__main__":
    main()
