"""
맨 처음 DB 만들기 위해 정의함
Author:
"""
import json
from sqlalchemy.orm import sessionmaker
from database import engine, Base
from models import Novel, Episode


def make_novel_db():
    novels = json.load(open("json/novel_info_list.json", encoding='utf-8'))
    Novel.__table__.create(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    for data in novels:
        novel_add = Novel(
            novel_id=data["id"],
            title=data["title"],
            writer=data["writer"],
            genre=data["genre"],
            total_epi=data["total_epi"],
            labeled=data["labeled"])
        session.add(novel_add)
        session.commit()

def make_episode_db():
    episodes = json.load(open("json/data.json", encoding='utf-8'))
    Episode.__table__.create(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    for episode in episodes:
        episode_add = Episode(
            novel_id=episode["id"],
            epi_title=episode["epi_title"],
            epi=episode["epi"],
            contents=episode["contents"])

        session.add(episode_add)
        session.commit()

if __name__ == "__main__":
    make_novel_db()
    make_episode_db()
