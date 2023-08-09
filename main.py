import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from timeit import default_timer as timer

# AES 256 encryption/decryption using pycryptodome library
# https://www.quickprogrammingtips.com/python/aes-256-encryption-and-decryption-in-python.htmlimport base64
from Crypto.Cipher import AES
from Crypto import Random
from Crypto.Protocol.KDF import PBKDF2

# variables for aes256 encryption
BLOCK_SIZE = 16
pad = lambda s: s + bytes((BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE), encoding="utf8")
unpad = lambda s: s[:-ord(s[len(s) - 1:])]
password = "aespassword"

def aes_get_private_key(password):
    salt = b"this is a salt"
    kdf = PBKDF2(password, salt, 64, 1000)
    key = kdf[:32]
    return key

def aes_encrypt(raw, password):
    private_key = aes_get_private_key(password)
    raw = pad(raw)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return base64.b64encode(iv + cipher.encrypt(raw))

def aes_decrypt(enc, password):
    private_key = aes_get_private_key(password)
    enc = base64.b64decode(enc)
    iv = enc[:16]
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc[16:]))

def run_mat_mul(n: int, m: int, scale: int) -> float:
    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    start = timer()
    res = a @ b
    stop = timer()
    return stop - start

def run_mat_mul_aes(n: int, m: int, scale: int) -> float:
    a = np.random.random((n, m)) * scale
    b = np.random.random((m, n)) * scale
    start = timer()
    
    # encrypting matrices, a and b
    a_encrypt = aes_encrypt(a.tobytes(), password)
    b_encrypt = aes_encrypt(b.tobytes(), password)

    # decrypting encrypted matrices
    a_decrypt = np.frombuffer(aes_decrypt(a_encrypt, password))
    a_decrypt.resize((n, m)) 
    b_decrypt = np.frombuffer(aes_decrypt(b_encrypt, password))
    b_decrypt.resize((m, n))

    res = a_decrypt @ b_decrypt

    stop = timer()
    return stop - start

def run_mat_mul_fhe(n: int, m: int, scale: int) -> float:
    HE = Pyfhel()
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

def run_logistic_aes() -> float:
    data, target = datasets.load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.5, random_state=None)
   
    # preprocess data
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    lin_reg = LogisticRegression()

    # run inference
    start = timer()

    # encrypting then decrypting data
    data_train_encrypt = aes_encrypt(data_train, password)
    data_test_encrypt = aes_encrypt(data_test, password)
    target_train_encrypt = aes_encrypt(target_train, password)

    data_train_decrypt = aes_decrypt(data_train_encrypt, password)
    data_test_decrypt = aes_decrypt(data_test_encrypt, password)
    target_train_decrypt = aes_decrypt(target_train_encrypt, password)

    y_pred = lin_reg.fit(data_train_decrypt, target_train_decrypt).predict(data_test_decrypt)
    stop = timer()
    print("Accuracy:", metrics.accuracy_score(target_test, y_pred))
    return stop - start


def run_logistic_reg_fhe() -> float:
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




if __name__ == "__main__":
    
    runs = 10
    time = 0.0
    for _ in range(0, runs):
        time += run_logistic_reg_fhe()
    print(f'FHE Logistic Regression Average Execution Time ({runs} runs): {time / runs}')

    '''
    time = 0.0
    for _ in range(0, runs):
        time += run_logistic_aes()
    print(f'AES256 Logistic Regression Average Execution Time ({runs} runs): {time / runs}')
    '''

    time = 0.0
    for _ in range(0, runs):
        time += run_logistic_reg()
    print(f'Standard Logistic Regression Average Execution Time ({runs} runs): {time / runs}')
    

    runs = 100
    time = 0.0
    for _ in range(0, runs):
        time += run_mat_mul(5, 5, 10)
    print(f'Standard Matrix Multiplication (5x5) Average Execution Time ({runs} runs): {time / runs}')

    time = 0.0
    for _ in range(0, runs):
        time += run_mat_mul_aes(5, 5, 10)
    print(f'AES-256 Matrix Multiplication (5x5) Average Execution Time ({runs} runs): {time / runs}')

    time = 0.0
    for _ in range(0, runs):
        time += run_mat_mul_fhe(5, 5, 10)
    print(f'FHE Matrix Multiplication (5x5) Average Execution Time ({runs} runs): {time / runs}')

    