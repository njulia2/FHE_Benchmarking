# AES 256 encryption/decryption using pycryptodome library
# https://www.quickprogrammingtips.com/python/aes-256-encryption-and-decryption-in-python.html
import base64
from Crypto.Cipher import AES
from Crypto import Random
from Crypto.Protocol.KDF import PBKDF2
import numpy as np
from Pyfhel import Pyfhel, PyCtxt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from timeit import default_timer as timer

BLOCK_SIZE = 16
pad = lambda s: s + bytes((BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE), encoding="utf8")
unpad = lambda s: s[:-ord(s[len(s) - 1:])]
password = "aespassword"

def get_private_key(password):
    salt = b"this is a salt"
    kdf = PBKDF2(password, salt, 64, 1000)
    key = kdf[:32]
    return key

def aes_encrypt(raw, password):
    private_key = get_private_key(password)
    raw = pad(raw)
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return base64.b64encode(iv + cipher.encrypt(raw))

def aes_decrypt(enc, password):
    private_key = get_private_key(password)
    enc = base64.b64decode(enc)
    iv = enc[:16]
    cipher = AES.new(private_key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(enc[16:]))

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

runs = 100
start_time = 0.0
for _ in range(0, runs):
    start_time += run_mat_mul_aes(5, 5, 10)
print(f'AES-256 Matrix Multiplication (5x5) Average Execution Time ({runs} runs): {start_time / runs}')