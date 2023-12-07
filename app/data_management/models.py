from sqlalchemy import Column, Integer, String
from sqlalchemy import ForeignKey, Text, Sequence
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Title(Base):
    __tablename__ = 'titles'
    id = Column(Integer, Sequence('title_id_seq'), primary_key=True,
                autoincrement=True)
    title_text = Column(String, unique=True, nullable=False)
    stories = relationship('Story', back_populates='title')
    namelists = relationship('NameList', back_populates='title')

class NameList(Base):
    __tablename__ = 'namelists'
    index = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(Integer, autoincrement=True)
    gender = Column(Integer)
    names = relationship("Name", back_populates="namelist")
    title_id = Column(Integer, ForeignKey('titles.id'))
    title = relationship('Title', back_populates='namelists')

class Name(Base):
    __tablename__ = 'names'
    id = Column(Integer, Sequence('name_id_seq'), primary_key=True,
                autoincrement=True)
    name = Column(String(50))
    namelist_id = Column(Integer, ForeignKey('namelists.id'))
    namelist = relationship("NameList", back_populates="names")

class Story(Base):
    __tablename__ = 'stories'
    id = Column(Integer, Sequence('story_id_seq'), primary_key=True,
                autoincrement=True)
    title_id = Column(Integer, ForeignKey('titles.id'), nullable=False)
    title = relationship('Title', back_populates='stories')
    instance_index = Column(Integer, nullable=False)
    instance = Column(Text, nullable=False)
    speaker = Column(String)
    speaker_index = Column(String)
    category = Column(String)
