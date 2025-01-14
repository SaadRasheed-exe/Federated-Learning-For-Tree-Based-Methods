import requests
import json
from abc import ABC, abstractmethod
from .Serializer import Serializer
from .BaseClientManager import BaseClientManager

class BaseServer(ABC):

    def __init__(self, clients_json_path: str):
        
        with open(clients_json_path, 'r') as f:
            clients = json.load(f)

        self.clients = {}
        for client_info in clients:
            ip = client_info['ip'].strip()
            state = client_info['state'].strip()

            self.clients[state] = ip

        self.serializer = Serializer()
        self.client_manager = BaseClientManager(self.clients, self.serializer)
    
    def _get_client_url(self, state='', ip=''):
        if state:
            return f'https://{self.clients[state]}:{self.client_manager.CLIENT_PORT}'
        elif ip:
            return f'https://{ip}:{self.client_manager.CLIENT_PORT}'
        else:
            return None
    
    def _get_client_data_stats(self, state):
        client_url = self._get_client_url(state=state)
        response = requests.get(f"{client_url}/send-stats", verify=False)
        return response.json()
        
    def get_data_stats(self, participants=None):
        if participants is None:
            participants = self.clients.keys()
        data_stats = {}
        for state in participants:
            if self._check_client_status(state) == 200:
                data_stats[state] = self._get_client_data_stats(state)
        return data_stats

    def _check_client_status(self, state='', ip=''):
        if not state and not ip:
            raise ValueError('Either state or ip should be provided.')
        try:
            client_url = self._get_client_url(state=state)
            response = requests.get(f"{client_url}/status", timeout=5, verify=False)
            return response.status_code
        except Exception as e:
            return {'message': f'Error: {e}'}
    
    def check_clients_status(self):
        statuses = {}
        for state in self.clients.keys():
            statuses[state] = self._check_client_status(state)
        return statuses

    
    def send_code_dir(self, code_dir: str, to: str = None):
        self.client_manager.send_code_dir(code_dir, to)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    