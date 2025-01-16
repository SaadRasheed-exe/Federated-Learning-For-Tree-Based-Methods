import requests
import json
from .Serializer import Serializer
from .BaseClientManager import BaseClientManager

class BaseServer:
    """
    A base class representing a server that manages multiple clients. It provides functionality for retrieving
    client status, statistics, and sending code directories to clients. The server interacts with the clients using 
    a `BaseClientManager` for communication and serialization.

    Attributes
    ----------
    clients : dict
        A dictionary mapping client states to their respective IP addresses.
    serializer : Serializer
        An instance of the `Serializer` class for serializing and deserializing messages.
    client_manager : BaseClientManager
        An instance of the `BaseClientManager` for managing communication with clients.

    Methods
    -------
    get_data_stats(participants: list, optional):
        Retrieves data statistics for one or more clients.

    check_clients_status() -> dict:
        Checks the status of all clients and returns a dictionary of client states and their statuses.

    send_code_dir(code_dir: str, to: str = None):
        Sends a directory of code to a client or multiple clients.

    fit() -> None:
        Abstract method to fit a model or execute training, to be implemented by subclasses.

    evaluate() -> None:
        Abstract method to evaluate a model, to be implemented by subclasses.
    """
    
    def __init__(self, clients_json_path: str):
        """
        Initializes the BaseServer with the list of clients from a JSON file.

        Parameters
        ----------
        clients_json_path : str
            The path to the JSON file containing the client information (IP and state).
        """
        with open(clients_json_path, 'r') as f:
            clients = json.load(f)

        self.clients = {}
        for client_info in clients:
            ip = client_info['ip'].strip()
            state = client_info['state'].strip()

            self.clients[state] = ip

        self.serializer = Serializer()
        self.client_manager = BaseClientManager(self.clients)
    
    def _get_client_url(self, state: str = '', ip: str = '') -> str:
        """
        Constructs the URL of the client based on either the state or IP address.

        Parameters
        ----------
        state : str, optional
            The state of the client to get the URL for.
        ip : str, optional
            The IP address of the client to get the URL for.

        Returns
        -------
        str
            The constructed client URL.
        
        Raises
        ------
        ValueError
            If neither state nor IP is provided.
        """
        if state:
            return f'https://{self.clients[state]}:{self.client_manager.CLIENT_PORT}'
        elif ip:
            return f'https://{ip}:{self.client_manager.CLIENT_PORT}'
        else:
            raise ValueError('Either state or ip should be provided.')
    
    def _get_client_data_stats(self, state: str) -> dict:
        """
        Retrieves data statistics from a client based on the client's state.

        Parameters
        ----------
        state : str
            The state of the client to retrieve data statistics from.

        Returns
        -------
        dict
            The data statistics from the client.
        """
        client_url = self._get_client_url(state=state)
        response = requests.get(f"{client_url}/send-stats", verify=False)
        return response.json()
        
    def get_data_stats(self, participants: list = None) -> dict:
        """
        Retrieves data statistics for one or more clients.

        Parameters
        ----------
        participants : list, optional
            A list of client states to retrieve data statistics for. If None, retrieves stats for all clients.

        Returns
        -------
        dict
            A dictionary of client states and their corresponding data statistics.
        """
        if participants is None:
            participants = self.clients.keys()
        data_stats = {}
        for state in participants:
            if self._check_client_status(state) == 200:
                data_stats[state] = self._get_client_data_stats(state)
        return data_stats

    def _check_client_status(self, state: str = '', ip: str = '') -> int:
        """
        Checks the status of a client by sending a GET request to the client's `/status` endpoint.

        Parameters
        ----------
        state : str, optional
            The state of the client to check the status for.
        ip : str, optional
            The IP address of the client to check the status for.

        Returns
        -------
        int
            The HTTP status code returned by the client. If an error occurs, returns a dictionary with the error message.
        
        Raises
        ------
        ValueError
            If neither state nor IP is provided.
        """
        if not state and not ip:
            raise ValueError('Either state or ip should be provided.')
        try:
            client_url = self._get_client_url(state=state)
            response = requests.get(f"{client_url}/status", timeout=5, verify=False)
            return response.status_code
        except Exception as e:
            return {'message': f'Error: {e}'}
    
    def check_clients_status(self) -> dict:
        """
        Checks the status of all clients and returns a dictionary mapping client states to their status codes.

        Returns
        -------
        dict
            A dictionary mapping client states to their status codes.
        """
        statuses = {}
        for state in self.clients.keys():
            statuses[state] = self._check_client_status(state)
        return statuses

    def send_code_dir(self, code_dir: str, to: str = None):
        """
        Sends a directory of code to a specific client or all active clients.

        Parameters
        ----------
        code_dir : str
            The path to the code directory to send.
        to : str, optional
            The client state to send the code to. If not provided, sends to all clients.
        """
        self.client_manager.send_code_dir(code_dir, to)
