from .EncryptionManager import EncryptionManager

class BaseClient:

    def __init__(self):
        self.encryption_manager = None
        self.server_public_key = None
    
    def init_encryption(self, server_public_key):
        self.encryption_manager = EncryptionManager()
        self.server_public_key = server_public_key
        return self.encryption_manager.public_key
    
    def encrypt_message(self, message):
        if self.encryption_manager:
            encrypted = self.encryption_manager.encrypt_message(self.server_public_key, message)
            return encrypted
        else:
            raise ValueError("Encryption manager not initialized.")

    def decrypt_message(self, message):
        if self.encryption_manager:
            decrypted = self.encryption_manager.decrypt_message(message)
            return decrypted
        else:
            raise ValueError("Encryption manager not initialized.")