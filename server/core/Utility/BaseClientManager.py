from .ServerEncryptionManager import ServerEncryptionManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, shutil
import requests
from typing import Dict, Any, Callable



class BaseClientManager:

    CLIENT_PORT = 7223

    def __init__(self, clients: Dict, encryption_manager: 'ServerEncryptionManager'):
        """
        Initializes the ClientManager instance.
        Args:
            clients (dict): A dictionary with client IDs as keys and their base URLs as values.
            encryption_manager (ServerEncryptionManager): The encryption manager instance to handle encryption/decryption.
        """
        self.clients = clients  # {client_id: client_url}
        self.encryption_manager = encryption_manager
        self.active_clients = list(self.clients.keys())  # List of active clients
        self.client_data = {}
    
    def _communicate(self, client_id: str, endpoint: str, data: Any = None, encrypt: bool = True) -> Dict:
        """
        Communicate with a client.
        Args:
            client_id (str): The ID of the client to send the request to.
            endpoint (str): The API endpoint to send the request to.
            data (any): The data to send with the request (optional).
        Returns:
            response (dict): The response from the client (if successful).
        """
        url = 'http://' + self.clients.get(client_id) + f':{self.CLIENT_PORT}'
        if not url:
            raise ValueError(f"Client ID {client_id} not found.")
        
        # Encrypt the data before sending (if data exists and encryption is enabled)
        if data and encrypt:
            data = {'encrypted': self.encryption_manager.encrypt_message(
                self.encryption_manager.get_client_public_key(client_id),
                data
            ).hex()}
        else:
            data = {'data': data}

        try:
            # Sending request to the client's API endpoint
            response = requests.post(f"{url}/{endpoint}", json=data)
            
            # Check if the response is successful
            if response.status_code == 200:
                encrypted = response.json().get('encrypted')
                if encrypted:
                    # Decrypt the received message
                    response = self.encryption_manager.decrypt_message(bytes.fromhex(encrypted))
                    return response
                else:
                    return response.json()
            else:
                # raise Exception(f"Failed to send request to {client_id}, Status Code: {response.status_code}")
                response.raise_for_status()
        
        except requests.exceptions.RequestException as e:
            print(f"Request sent at {endpoint} failed for {client_id}: {str(e)}")
            self._handle_client_failure(client_id)
            return None
    
    def _execute_in_threads(self, max_workers: int, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes a function concurrently for all active clients.
        Args:
            func (Callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        Returns:
            results (dict): A dictionary with client IDs as keys and function results as values.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map each client ID to a thread running the function
            future_to_client = {
                executor.submit(func, client_id, *args, **kwargs): client_id
                for client_id in self.active_clients
            }
            for future in as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    results[client_id] = future.result()
                except Exception as e:
                    print(f"Error executing function for client {client_id}: {str(e)}")
                    self._handle_client_failure(client_id)
        return results
    
    # Function to upload a file to a specific subdirectory on the client
    def _upload_to_client(self, client_id, path, their_subdir=''):
        url = 'http://' + self.clients.get(client_id) + f':{self.CLIENT_PORT}'
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")
        
        # Determine if it's a file or a folder
        if os.path.isfile(path):
            upload_path = path
            filename = os.path.basename(upload_path)
            data = {'subdirectory': their_subdir, 'format': 'file'}
        elif os.path.isdir(path):
            # Compress the folder into a .zip file
            upload_path = shutil.make_archive(path, 'zip', path)
            filename = os.path.basename(upload_path)
            data = {'subdirectory': their_subdir, 'format': 'zip'}
        else:
            raise ValueError("The specified path is neither a file nor a folder.")
        
        # Open the file and upload it
        with open(upload_path, 'rb') as f:
            files = {'file': (filename, f)}
            response = requests.post(f"{url}/upload", files=files, data=data)
        
        # Clean up the temporary .zip file if a folder was compressed
        if os.path.isdir(path):
            os.remove(upload_path)
        
        return response
    
    def _handle_client_failure(self, client_id: str):
        """
        Handle client failure (remove from the active list).
        Args:
            client_id (str): The ID of the failed client.
        """
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)
            print(f"Client {client_id} has been removed from active clients.")
    
    def init_encryption(self, endpoint: str):
        """
        Initialize encryption for all active clients.
        """
        server_public_key = self.encryption_manager.get_public_key()
        data = {'server_public_key': server_public_key.hex()}
        result = self._execute_in_threads(4, self._communicate, f'{endpoint}/init-encryption', data, encrypt=False)
        for client_id, response in result.items():
            if response:
                self.encryption_manager.add_client_public_key(client_id, bytes.fromhex(response.get('client_public_key')))
    
    def manage_active_clients(self):
        """
        Periodically check the status of active clients, remove any that have failed.
        """
        self._execute_in_threads(4, self._communicate, 'status', encrypt=False)

    def get_client_public_key(self, client_id: str):
        """
        Retrieve the public key of a client by their client ID.
        Args:
            client_id (str): The ID of the client.
        Returns:
            public_key (bytes): The public key in PEM format.
        """
        return self.encryption_manager.get_client_public_key(client_id)
    
    def send_code_dir(self, code_dir: str):
        """
        Send the code directory to all active clients.
        Args:
            code_dir (str): The path to the code directory.
        """
        self._execute_in_threads(1, self._upload_to_client, code_dir)