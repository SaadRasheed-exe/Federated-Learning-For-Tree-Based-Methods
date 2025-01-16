from .Serializer import Serializer
from ..Models import MajorityVotingEnsemble

class BaseClient:

    def __init__(self):
        self.serializer = Serializer()
    
    def serialize_message(self, message):
        if self.serializer:
            serialized = self.serializer.serialize_message(message)
            return serialized
        else:
            raise ValueError("Serializer not initialized.")

    def deserialize_message(self, message):
        if self.serializer:
            deserialized = self.serializer.deserialize_message(message)
            return deserialized
        else:
            raise ValueError("Serializer not initialized.")