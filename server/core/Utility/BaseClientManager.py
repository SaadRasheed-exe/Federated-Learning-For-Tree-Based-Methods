from .Serializer import Serializer
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, shutil
import requests
from typing import Dict, Any, Callable

class BaseClientManager:
    """
    A class responsible for managing communication with multiple clients, enabling operations like sending data 
    and uploading files to clients. This class supports concurrent communication using threads.

    Attributes
    ----------
    CLIENT_PORT : int
        The port on which clients are running.
    clients : Dict[str, str]
        A dictionary mapping client IDs to client URLs.
    serializer : Serializer
        A Serializer object for serializing and deserializing messages.
    active_clients : list
        A list of active client IDs.
    client_data : dict
        A dictionary to hold data related to clients.

    Methods
    -------
    send_code_dir(code_dir: str, to: str = None):
        Sends a directory containing code to a client or multiple clients.
    """
    
    CLIENT_PORT = 7223

    def __init__(self, clients: Dict, serializer: 'Serializer'):
        """
        Initializes the client manager with a list of clients and a serializer.

        Parameters
        ----------
        clients : dict
            A dictionary mapping client IDs to client URLs.
        serializer : Serializer
            A Serializer object for serializing and deserializing messages.
        """
        self.clients = clients  # {client_id: client_url}
        self.serializer = serializer
        self.active_clients = list(self.clients.keys())  # List of active clients
        self.client_data = {}

    def _communicate(self, client_id: str, endpoint: str, data: Any = None, serialize: bool = True) -> Dict:
        """
        Sends data to a client and returns the response from the client.

        Parameters
        ----------
        client_id : str
            The ID of the client to communicate with.
        endpoint : str
            The endpoint on the client's server to send the data to.
        data : any, optional
            The data to send to the client.
        serialize : bool, default=True
            Whether to serialize the data before sending.

        Returns
        -------
        dict
            The response from the client.
        
        Raises
        ------
        ValueError
            If the client ID is not found in the list of clients.
        requests.exceptions.RequestException
            If the request fails due to connection or status code issues.
        """
        url = 'https://' + self.clients.get(client_id) + f':{self.CLIENT_PORT}'
        if not url:
            raise ValueError(f"Client ID {client_id} not found.")
        
        # Serialize the data before sending (if data exists and serialization is enabled)
        if data and serialize:
            data = {'serialized': self.serializer.serialize_message(data).hex()}
        else:
            data = {'data': data}

        try:
            # Sending request to the client's API endpoint
            response = requests.post(f"{url}/{endpoint}", json=data, verify=False)
            
            # Check if the response is successful
            if response.status_code == 200:
                serialized = response.json().get('serialized')
                if serialized:
                    # Deserialize the received message
                    response = self.serializer.deserialize_message(bytes.fromhex(serialized))
                    return response
                else:
                    return response.json()
            else:
                response.raise_for_status()
        
        except requests.exceptions.RequestException as e:
            print(f"Request sent at {endpoint} failed for {client_id}: {str(e)}")
            self._handle_client_failure(client_id)
            return None

    def _execute_in_threads(self, max_workers: int, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes a function concurrently across multiple clients using threads.

        Parameters
        ----------
        max_workers : int
            The maximum number of workers (threads) to use for parallel execution.
        func : Callable
            The function to execute on each client.
        *args : additional arguments
            Arguments to pass to the function.
        **kwargs : additional keyword arguments
            Keyword arguments to pass to the function.

        Returns
        -------
        dict
            A dictionary mapping client IDs to their respective results.
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

    def _upload_to_client(self, client_id: str, path: str, their_subdir: str = '') -> requests.Response:
        """
        Uploads a file or folder to a specific subdirectory on the client.

        Parameters
        ----------
        client_id : str
            The ID of the client to upload the file to.
        path : str
            The path of the file or folder to upload.
        their_subdir : str, default=''
            The subdirectory on the client where the file should be uploaded.

        Returns
        -------
        requests.Response
            The response from the client's server.
        
        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.
        ValueError
            If the specified path is neither a file nor a folder.
        """
        url = 'https://' + self.clients.get(client_id) + f':{self.CLIENT_PORT}'
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
            response = requests.post(f"{url}/upload", files=files, data=data, verify=False)
        
        # Clean up the temporary .zip file if a folder was compressed
        if os.path.isdir(path):
            os.remove(upload_path)
        
        return response

    def _handle_client_failure(self, client_id: str):
        """
        Handles a client failure by removing the client from the active clients list.

        Parameters
        ----------
        client_id : str
            The ID of the failed client.
        """
        if client_id in self.active_clients:
            self.active_clients.remove(client_id)
            print(f"Client {client_id} has been removed from active clients.")

    def send_code_dir(self, code_dir: str, to: str = None):
        """
        Sends a directory containing code to a specific client or all active clients.

        Parameters
        ----------
        code_dir : str
            The path to the code directory to send.
        to : str, optional
            The specific client to send the code to. If not provided, sends to all active clients.
        """
        if to:
            self._upload_to_client(to, code_dir)
        else:
            self._execute_in_threads(1, self._upload_to_client, code_dir)
