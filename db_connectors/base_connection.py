import logging

class BaseConnectionHandler:
    def __init__(self, config, logger=None):
        """
        Base DB connection handler class with a configuration dictionary.

        :param config: A dictionary containing database connection details.
        :param logger: A logger instance
        """
        self.db_name = config.get("db_name")
        self.db_schema = config.get("db_schema")
        self.db_user = config.get("db_user")
        self.db_password = config.get("db_password")
        self.db_host = config.get("db_host")
        self.db_port = config.get("db_port")
        self.min_pool_size = config.get("min_pool_size", 20)
        self.max_pool_size = config.get("max_pool_size", 30)
        self.pool = None
        self.logger = logger or logging.getLogger(__name__)


    def close_pool(self):
        """
        Closes the connection pool. Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")