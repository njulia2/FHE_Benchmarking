import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from timeit import default_timer as timer
import argparse 
import json


def run_mat_mulf(n: int, m: int, scale: int) -> float:
    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    start = timer()
    res = a @ b
    stop = timer()
    return stop - start


def run_mat_muli(n: int, m: int, scale: int) -> float:
    a = np.random.randint(scale, size=(n, m))
    b = np.random.randint(scale, size=(m, n))
    start = timer()
    res = a @ b
    stop = timer()
    return stop - start



def run_mat_muli_fhe(n: int, m: int, scale: int, params=None) -> float:
    HE = Pyfhel()
    if params is not None: 
        bfv_params = params
    else:
        bfv_params = {
            'scheme': 'BFV',
            'n': 2 ** 13,
            't': 65537,
            't_bits': 20,
            'sec': 128,
        }
    HE.contextGen(**bfv_params)
    HE.keyGen()
    HE.rotateKeyGen()
    HE.relinKeyGen()
    a = np.random.randint(scale, size=(n, m))
    b = np.random.randint(scale, size=(m, n))
    a_enc = [HE.encryptInt(np.array(row)) for row in a]
    b_enc = [HE.encryptInt(np.array(col)) for col in b.T]
    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)

    stop = timer()

    return stop - start


def run_mat_mulf_fhe(n: int, m: int, scale: int, params=None) -> float:
    HE = Pyfhel()
    if params is not None: 
        ckks_params = params
    else:
        ckks_params = {
            "scheme": "CKKS",
            "n": 2 ** 14,
            "scale": 2 ** 30,
            "qi_sizes": [60, 30, 30, 30, 60]
        }
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    a_mat = np.random.random((n, m)) * scale
    b_mat = (np.random.random((m, n)) * scale)
    a_enc = [HE.encryptFrac(np.array(row)) for row in a_mat]
    b_enc = [HE.encryptFrac(np.array(col)) for col in b_mat.T]
    start = timer()
    res = []
    for a_row in a_enc:
        sub_res = []
        for b_col in b_enc:
            sub_res.append(HE.scalar_prod(a_row, b_col, in_new_ctxt=True))
        res.append(sub_res)
    stop = timer()
    return stop - start


def run_logistic_reg() -> float:
    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    # run inference
    start = timer()
    y_pred = lin_reg.fit(data_train, target_train).predict(data_test)
    stop = timer()
    print("Accuracy:", metrics.accuracy_score(target_test, y_pred))
    return stop - start


def run_logistic_reg_fhe(params=None) -> float:
    data, target = datasets.load_iris(return_X_y=True)

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()
    lin_reg.fit(data_train, target_train)
    # encrypt test data
    HE = Pyfhel()
    if params is not None: 
        ckks_params = params
    else:
        ckks_params = {
            "scheme": "CKKS",
            "n": 2 ** 14,
            "scale": 2 ** 30,
            "qi_sizes": [60, 30, 30, 30, 60]
        }
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    data_appended = np.append(data_test, np.ones((len(data_test), 1)), -1)
    encrypted_test_data = [HE.encryptFrac(row) for row in data_appended]

    # encrypt coefficients from trained model, start of inferencing process
    start = timer()
    coefs = []
    for i in range(0, 3):
        coefs.append(np.append(lin_reg.coef_[i], lin_reg.intercept_[i]))
    encoded_coefs = [HE.encodeFrac(coef) for coef in coefs]
    # run inference
    predictions = []
    for data in encrypted_test_data:
        encrypted_prediction = [
            HE.scalar_prod_plain(data, encoded_coef, in_new_ctxt=True)
            for encoded_coef in encoded_coefs
        ]
        predictions.append(encrypted_prediction)
    stop = timer()
    # decrypt predictions and check accuracy
    c1_preds = []
    for prediction in predictions:
        cl = np.argmax([HE.decryptFrac(logit)[0] for logit in prediction])
        c1_preds.append(cl)
    print("Accuracy:", metrics.accuracy_score(target_test, c1_preds))
    return stop - start


def main(args):
    test_desc = json.load(open(args.file_input))    
    with open(test_desc["out_file"], "w") as logger:
        # iterate over list of tests in test file
        cnt = 0
        for test in test_desc["tests"]:
            logger.write(f'Running Test: {cnt}, {test["desc"]}\n')
            func = None 
            args = [] 
            if test["type"] == "log_reg": 
                func = run_logistic_reg_fhe if test["encryption"] == "FHE" else run_logistic_reg 

            elif test["type"] == "mat_mul": 
                if test["encryption"] == "FHE" and test["data_type"] == "float": 
                    func = run_mat_mulf_fhe 
                elif test["encryption"] == "FHE" and test["data_type"] == "int":
                    func = run_mat_muli_fhe 
                elif test["encryption"] == "NONE" and test["data_type"] == "float":
                    func = run_mat_mulf
                elif test["encryption"] == "NONE" and test["data_type"] == "int": 
                    func = run_mat_muli 
                else: 
                    print('Invalid Test Case')
                    continue
                args = [test["mat_size"][0], test["mat_size"][1], test["scale"]]
            else: 
                print(f'{test["type"]} Not a supported test type')
                continue 

            # if the file defines pyfhel params, use them 
            if  "pyfhel_params" in test:
                args += [test["pyfhel_params"]]
             # run the test
            time = 0.0
            runs = test["runs"]
            logger.write(f'Using Args: {args} ')
            for _ in range(runs):
                time += func(*args)
            logger.write(f'Average Execution Time (Runs {runs}) : {time / runs}\n')
            cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Bench Tool for comparing FHE, AES, and Non-encrypted operations', fromfile_prefix_chars='@')
    parser.add_argument('--file_input', '-f', help='Test Description File: takes a string to a test description file, expects a JSON file describing tests to be run')
    parser.add_argument('--test_type', '-t', help='Test Type: takes a string denoting the type of test to perform (mat_mul, mat_scale, log_reg, add, sub, mult, div)')
    parser.add_argument('--encryption', '-e', help='Encryption Type: takes a string denoting the type of encryption to use (FHE, AES, NONE)')
    parser.add_argument('--data_type', '-d', help='Data Type: takes a string denoting the type of data used during the test (float, int)')
    parser.add_argument('--runs', '-r', help='Test Runs: takes an integer number of iterations to run the test', default=10)
    parser.add_argument('--scale', '-s', help='Data Scale: Scale factor applied to randomly generated data, not applicable to log_reg test', default=1)
    parser.add_argument('--mat_size', '-m', help='Matrix Size: takes an integer tuple (n, m) denoting the size of the matrix to be tested with. Only applicable to mat_mul and mat_scale tests', default=(5, 5))
    parser.add_argument('--out_file', '-o', help='Outfile Name: takes a string denoting the name of the outfile to store test results in', default='results.txt')
    main(parser.parse_args())
