"""
This is crud.py
"""
from sqlalchemy import inspect

from .database import engine


def show_table_info(table_name='all'):
    """
    데이터베이스의 테이블 정보(테이블 이름, 열 이름, 데이터 유형, null 허용 여부)를 출력합니다.
    """
    # SQLAlchemy Inspector 생성
    inspector = inspect(engine)

    # 모든 테이블의 정보 가져오기
    if table_name == 'all':
        tables = inspector.get_table_names()

        for table_names in tables:
            print(f"\nTable: {table_names}")
            columns = inspector.get_columns(table_names)
            for column in columns:
                print(f"Column: {column['name']}, Type: {column['type']}, "
                      f"Nullable: {column['nullable']}")

    # 지정된 테이블의 정보 가져오기
    else:
        columns = inspector.get_columns(table_name)

        print(f"Table: {table_name}")
        for column in columns:
            print(f"Column: {column['name']}, Type: {column['type']}, "
                  f"Nullable: {column['nullable']}")
