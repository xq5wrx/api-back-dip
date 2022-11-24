import pymysql.cursors
from pymysql.cursors import DictCursor
from pymysql.constants import CLIENT


# Функция возвращает connection.
def getConnection():
    # Вы можете изменить параметры соединения.
    return pymysql.connect(
        host='db',
        port=3306,
        db='dip',
        user='root',
        password='root',
        charset='utf8mb4',
        cursorclass=DictCursor,
        client_flag=CLIENT.MULTI_STATEMENTS
    )
