from sqlalchemy import Column, Integer, String, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class Novel(Base):
    __tablename__ = "novels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    novel_id = Column(Integer, nullable=False)
    title = Column(String)
    writer = Column(String)
    genre = Column(String)
    total_epi = Column(Integer)
    labeled = Column(Boolean)

    episodes = relationship("Episode", back_populates="novel", lazy='joined')

class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    novel_id = Column(Integer, ForeignKey("novels.novel_id"), nullable=False)
    epi_title = Column(String)
    epi = Column(Integer)
    contents = Column(JSON)

    novel = relationship("Novel", back_populates="episodes")
