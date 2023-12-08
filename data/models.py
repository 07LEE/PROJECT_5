"""
Author: H6
"""
# %%
# from sqlalchemy importString, ForeignKey,
from sqlalchemy import create_engine, Column, Integer, Text, Sequence
from sqlalchemy.orm import declarative_base, sessionmaker
# from sqlalchemy.orm import relationship

db_file_path = "mydatabase.db"
engine = create_engine(f'sqlite:///{db_file_path}', echo=True)
Base = declarative_base()
Session = sessionmaker(engine)
sess = Session()

class Title(Base):
    __tablename__ = 'titles'
    id = Column(Integer, Sequence('title_id_seq'), primary_key=True, autoincrement=True)
    title_text = Column('TITLE_TEXT', Text)

Base.metadata.create_all(engine)

# %%
