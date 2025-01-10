from .EncryptionManager import EncryptionManager

class ServerEncryptionManager(EncryptionManager):

    def __init__(self):
        super().__init__()
        self.client_public_keys = {}
    
    def add_client_public_key(self, client_id, public_key):
        """
        Add a client's public key to the server's list of client public keys.
        """
        self.client_public_keys[client_id] = public_key
    
    def get_client_public_key(self, client_id):
        """
        Retrieve a client's public key by their client ID.
        """
        return self.client_public_keys.get(client_id)