import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ..Models.agg import MajorityVotingEnsemble
from ..Models.fedxgb import FedXGBoostEnsemble


class Serializer:

    def serialize_message(self, message):
        """
        Serialize a message of any data type using pickle serialization.
        Args:
            message (any): The message to serialize (any data type).
        Returns:
            serialized_data (bytes): The serialized message.
        """
        # Serialize the message
        serialized_message = pickle.dumps(message)
        return serialized_message

    def deserialize_message(self, serialized_message):
        """
        Deserialize a message serialized with hybrid RSA and AES serialization.
        Args:
            serialized_message (bytes): The serialized data (RSA-serialized AES key + AES ciphertext).
        Returns:
            message (any): The deserialized original message.
        """

        # Deserialize the original message
        message = pickle.loads(serialized_message)
        return message