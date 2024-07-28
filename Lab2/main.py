import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Lab2.Adversarial import AdversarialExampleGenerator, AdversarialTraining, ModelRobustifying

# Redirect print outputs to a file
output_file = open("Results/output.txt", "w")
sys.stdout = output_file

np.random.seed(10)

# Define column names and numeric columns for the dataset
COL_NAME = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerrorate', 'srv_rerrorate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

NUMERIC_COLS = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerrorate',
                'srv_rerrorate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']


def main():
    print("<" * 15, "Initialization", ">" * 15)
    EPOCH = 50
    TEST_RATE = 0.2
    VALIDATION_RATE = 0.2

    # Fetch and preprocess the dataset
    X, y = get_ds()
    num_class = len(np.unique(y))

    # Define attack and defense functions
    attack_functions = [
        AdversarialExampleGenerator.generate_fgsm_attack,
        AdversarialExampleGenerator.generate_bim_attack,
        AdversarialExampleGenerator.generate_pgd_attack,
        AdversarialExampleGenerator.generate_mim_attack
    ]

    defense_functions = [
        AdversarialExampleGenerator.defense_gan,
        AdversarialExampleGenerator.input_reconstruction,
        AdversarialTraining.adversarial_training,
        ModelRobustifying.model_robustifying
    ]

    # Create and train the original model
    model = create_tf_model(X.shape[1], num_class)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATE)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    history = model.fit(X_train, y_train_cat, epochs=EPOCH, batch_size=50000, verbose=0, validation_split=VALIDATION_RATE)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm_org = confusion_matrix(y_test, y_pred)
    print("_" * 50)
    print("Original confusion matrix")
    print(cm_org)
    evaluation_metrics(y_test, y_pred)
    print("_" * 50)

    results = []

    # Perform attacks, apply defenses, and evaluate performance
    for attack_function in attack_functions:
        print("<" * 15, attack_function.__name__, ">" * 15)
        for epsilon in [0.1, 0.5, 1, 1.5, 2, 2.5]:
            print("-" * 35)
            print("eps: ", epsilon)
            model = create_tf_model(X.shape[1], num_class)
            history = model.fit(X_train, y_train_cat, epochs=EPOCH, batch_size=50000, verbose=0, validation_split=VALIDATION_RATE)
            adv_x = attack_function(AdversarialExampleGenerator(model, X.shape[1]), X_test, epsilon)

            y_pred = np.argmax(model.predict(adv_x), axis=1)
            cm_adv = confusion_matrix(y_test, y_pred)

            print("Attacked confusion matrix")
            print(cm_adv)
            eval_metrics = evaluation_metrics(y_test, y_pred)
            results.append(['None', attack_function.__name__, epsilon] + eval_metrics)

            for defense_function in defense_functions:
                print("Applying defense: ", defense_function.__name__)
                if defense_function == AdversarialTraining.adversarial_training:
                    defense_function(AdversarialTraining(model, X.shape[1]), attack_function, X_train, y_train_cat, epsilon, epochs=5)
                    defended_x = attack_function(AdversarialExampleGenerator(model, X.shape[1]), X_test, epsilon)
                elif defense_function == ModelRobustifying.model_robustifying:
                    defense_function(ModelRobustifying(model, X.shape[1]), X_train, y_train_cat, epsilon, iterations=10)
                    defended_x = attack_function(AdversarialExampleGenerator(model, X.shape[1]), X_test, epsilon)
                else:
                    defended_x = defense_function(AdversarialExampleGenerator(model, X.shape[1]), adv_x)

                y_pred = np.argmax(model.predict(defended_x), axis=1)
                cm_defended = confusion_matrix(y_test, y_pred)

                print("Defended confusion matrix")
                print(cm_defended)
                eval_metrics = evaluation_metrics(y_test, y_pred)
                results.append([defense_function.__name__, attack_function.__name__, epsilon] + eval_metrics)

    results_df = pd.DataFrame(results, columns=['Defense', 'Attack', 'Epsilon', 'Accuracy', 'Recall', 'Precision', 'F1 Score'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    print(results_df)

    output_file.close()


def evaluation_metrics(y_test, y_pred):
    """
    Calculate and print evaluation metrics for the model.

    Args:
        y_test (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        list: List of evaluation metrics [accuracy, recall, precision, f1].
    """
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("accuracy_score: ", accuracy)
    print("recall_score: ", recall)
    print("precision_score: ", precision)
    print("f1_score: ", f1)
    return [accuracy, recall, precision, f1]


def get_ds():
    """
    Fetch and preprocess the KDDCUP'99 dataset.

    Returns:
        numpy.ndarray: Preprocessed input data.
        numpy.ndarray: Preprocessed labels.
    """
    x_kddcup, y_kddcup = fetch_kddcup99(return_X_y=True, shuffle=False)
    df_kddcup = pd.DataFrame(x_kddcup, columns=COL_NAME)
    df_kddcup['label'] = y_kddcup
    df_kddcup.drop_duplicates(keep='first', inplace=True)
    df_kddcup['label'] = df_kddcup['label'].apply(lambda d: str(d).replace('.', '').replace("b'", "").replace("'", ""))

    conversion_dict = {'back': 'dos', 'buffer_overflow': 'u2r', 'ftp_write': 'r2l',
                       'guess_passwd': 'r2l', 'imap': 'r2l', 'ipsweep': 'probe',
                       'land': 'dos', 'loadmodule': 'u2r', 'multihop': 'r2l',
                       'neptune': 'dos', 'nmap': 'probe', 'perl': 'u2r', 'phf': 'r2l',
                       'pod': 'dos', 'portsweep': 'probe', 'rootkit': 'u2r',
                       'satan': 'probe', 'smurf': 'dos', 'spy': 'r2l', 'teardrop': 'dos',
                       'warezclient': 'r2l', 'warezmaster': 'r2l'}
    df_kddcup['label'] = df_kddcup['label'].replace(conversion_dict)
    df_kddcup.query("label != 'u2r'", inplace=True)
    df_y = pd.DataFrame(df_kddcup.label, columns=["label"], dtype="category")
    df_kddcup.drop(["label"], inplace=True, axis=1)
    x_kddcup = df_kddcup[NUMERIC_COLS].values
    x_kddcup = preprocessing.scale(x_kddcup)
    y_kddcup = df_y.label.cat.codes.to_numpy()
    return x_kddcup, y_kddcup


def create_tf_model(input_size, num_of_class):
    """
    Create and compile a TensorFlow classification model.

    Args:
        input_size (int): The size of the input features.
        num_of_class (int): The number of output classes.

    Returns:
        tf.keras.Sequential: The compiled TensorFlow model.
    """
    model_kddcup = tf.keras.Sequential([
        tf.keras.layers.Dense(200, input_dim=input_size, activation=tf.nn.relu),
        tf.keras.layers.Dense(500, activation=tf.nn.relu),
        tf.keras.layers.Dense(200, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_of_class),
        tf.keras.layers.Activation(tf.nn.softmax)
    ])
    model_kddcup.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_kddcup


if __name__ == '__main__':
    main()
