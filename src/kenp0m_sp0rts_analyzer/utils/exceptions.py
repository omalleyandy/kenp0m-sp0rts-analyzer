"""Custom exceptions for the application"""


class NCAABException(Exception):
    """Base exception for the application"""

    pass


class DataValidationError(NCAABException):
    """Raised when data validation fails"""

    pass


class FeatureEngineeringError(NCAABException):
    """Raised when feature engineering fails"""

    pass


class ModelError(NCAABException):
    """Raised when model operations fail"""

    pass


class PredictionError(NCAABException):
    """Raised when prediction fails"""

    pass


class DatabaseError(NCAABException):
    """Raised when database operations fail"""

    pass


class ConfigurationError(NCAABException):
    """Raised when configuration is invalid"""

    pass
