from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


class DBConnectionHandler:

    def __init__(self) -> None:
        # Carregar as variáveis de ambiente do arquivo .env
        load_dotenv()  # Isso irá carregar as variáveis do arquivo .env

        # Ler as variáveis de ambiente
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')
        db_name = os.getenv('DB_NAME')

        # Construir a string de conexão com os valores carregados
        self.__connection_string = "{}://{}:{}@{}:{}/{}".format(
            'mysql+pymysql',
            user,
            password,
            host,
            port,
            db_name
        )
        self.__engine = self.__create_database_engine()
        self.session = None

    def __create_database_engine(self):
        engine = create_engine(self.__connection_string)
        return engine

    def get_engine(self):
        return self.__engine