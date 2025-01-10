import os
import pickle
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class EncryptionManager:
    def __init__(self):
        self.private_key, self.public_key = self._generate_keys()

    def _generate_keys(self):
        """
        Generate a pair of RSA keys (private and public).
        Returns:
            private_key_pem (bytes): The private key in PEM format.
            public_key_pem (bytes): The public key in PEM format.
        """
        private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        )
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_key_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_key_pem, public_key_pem

    def get_public_key(self):
        """
        Retrieve the public key.
        Returns:
            public_key (bytes): The public key in PEM format.
        """
        return self.public_key

    def encrypt_message(self, key, message):
        """
        Encrypt a message of any data type using hybrid RSA and AES encryption.
        Args:
            key (bytes): The public key in PEM format.
            message (any): The message to encrypt (any data type).
        Returns:
            encrypted_data (bytes): The encrypted message (RSA-encrypted AES key + AES ciphertext).
        """
        # Serialize the message
        serialized_message = pickle.dumps(message)

        # Generate a random AES key
        aes_key = os.urandom(32)  # 256-bit AES key

        # Encrypt the serialized message with AES
        iv = os.urandom(16)  # Initialization vector
        cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(serialized_message) + encryptor.finalize()

        # Encrypt the AES key with RSA
        public_key = serialization.load_pem_public_key(key)
        encrypted_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Combine the encrypted AES key, IV, and ciphertext
        encrypted_data = pickle.dumps({
            "encrypted_key": encrypted_key,
            "iv": iv,
            "ciphertext": ciphertext
        })

        return encrypted_data

    def decrypt_message(self, message):
        """
        Decrypt a message encrypted with hybrid RSA and AES encryption.
        Args:
            key (bytes): The private key in PEM format.
            message (bytes): The encrypted data (RSA-encrypted AES key + AES ciphertext).
        Returns:
            message (any): The decrypted original message.
        """
        # Deserialize the encrypted data
        data = pickle.loads(message)
        encrypted_key = data["encrypted_key"]
        iv = data["iv"]
        ciphertext = data["ciphertext"]

        # Decrypt the AES key with RSA
        private_key = serialization.load_pem_private_key(self.private_key, password=None)
        aes_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Decrypt the ciphertext with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        serialized_message = decryptor.update(ciphertext) + decryptor.finalize()

        # Deserialize the original message
        message = pickle.loads(serialized_message)

        return message

    @staticmethod
    def save_key_to_file(key, file_name):
        """
        Save a key (private or public) to a file.
        Args:
            key (bytes): The key in PEM format.
            file_name (str): The file path to save the key.
        """
        with open(file_name, 'wb') as key_file:
            key_file.write(key)
    
    @staticmethod
    def load_key_from_file(filename):
        """
        Load a key (private or public) from a file.
        Args:
            filename (str): The file path to load the key from.
        Returns:
            key_pem (bytes): The key in PEM format.
        """
        with open(filename, 'rb') as key_file:
            return key_file.read()
