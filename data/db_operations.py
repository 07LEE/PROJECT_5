"""
Author: 
"""
import json
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import event

Base = declarative_base()
engine = create_engine('sqlite:///data/mydatabase.db', echo=False)
Session = sessionmaker(bind=engine)

# 새로운 NameList 객체가 추가될 때 호출되는 이벤트 핸들러
@event.listens_for(NameList, 'before_insert')
def receive_before_insert(mapper, connection, target):
    """
    
    """
    session = Session.object_session(target)
    if session is not None:
        # 현재 title_id에 해당하는 가장 높은 id 값 조회
        max_id = session.query(NameList.id).filter_by(
            title_id=target.title_id).order_by(NameList.id.desc()).first()
        if max_id:
            target.id = max_id[0] + 1
        else:
            target.id = 1


def add_namelist(new_name, name=False, gender=2, title_text=None):
    """
    name: 기존의 인물 사전에 있는 이름.
    new_name: 추가할 새로운 이름
    gender : 성별 (0: female, 1: male, 2: 그 외)
    title_text: 추가할 타이틀 텍스트
    """
    session = Session()
    try:
        # 데이터베이스의 기존 데이터를 가져옴
        namelist_data = session.query(NameList).join(Title).filter(
            NameList.names.any(Name.name == name),
            NameList.title_id == Title.id,
            Title.title_text == title_text).first()

        if namelist_data:
            # 새로운 이름 추가
            new_name_obj = Name(name=new_name)
            namelist_data.names.append(new_name_obj)

            # title text 추가
            if title_text:
                title = session.query(Title).filter_by(title_text=title_text).first()
                if title is None:
                    title = Title(title_text=title_text)
                    session.add(title)
                    session.commit()
                namelist_data.title = title

            session.commit()
            print(f"{new_name}이(가) {name}의 이름 목록에 추가되었습니다.")

        else:
            # 기존 namelist가 없는 경우, 새로운 namelist를 생성하고 이름 추가
            new_namelist = NameList(gender=gender)  # gender 미지정할 경우 2
            new_name_obj = Name(name=new_name)
            new_namelist.names.append(new_name_obj)

            if title_text:
                title = session.query(Title).filter_by(title_text=title_text).first()
                if title is None:
                    title = Title(title_text=title_text)
                    session.add(title)
                    session.commit()
                new_namelist.title = title

            session.add(new_namelist)
            session.commit()
            print(f"{new_name}이(가) 새로운 이름 목록에 추가되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        # 세션 종료
        session.close()


def print_namelist(name):
    """
    주어진 이름에 해당하는 NameList와 그에 속한 이름들을 출력합니다.
    name: 출력하고자 하는 이름
    """
    try:
        # 세션 열기
        with Session() as session:
            filtered_data = session.query(NameList).filter(
                NameList.names.any(Name.name == name)).all()

            # 출력
            print("\nFiltered Data:")
            for data in filtered_data:
                print(data.id, data.gender, [(name.id, name.name) for name in data.names])

    except Exception as e:
        print(f"오류 발생: {e}")


def display_namelist(save=False):
    """
    name list를 정해진 형식으로 보여줍니다.
    """
    try:
        # 세션 열기
        with Session() as session:
            # NameList 테이블의 모든 데이터 조회
            namelist_data = session.query(NameList).all()

            namelist_dict = {}
            for data in namelist_data:
                names_list = [name.name for name in data.names]
                namelist_dict[data.id] = [data.gender] + names_list
                print(data.id, data.gender, names_list)

            print("\nFormatted NameList Data:")
            formatted_namelist_dict = {}
            for key, value in namelist_dict.items():
                formatted_namelist_dict[key] = value

            print(formatted_namelist_dict)

            if save is True:
                with open('formatted_namelist_data.json', 'w') as json_file:
                    json.dump(formatted_namelist_dict, json_file, ensure_ascii=False, default=str,
                              indent=4)

    except Exception as e:
        print(f"오류 발생: {e}")


def json2db(file):
    """
    Json 소설파일을 db에 집어넣습니다.
    """
    # SQLAlchemy 세션 생성
    session = Session()

    # JSON 파일 읽고 데이터베이스에 데이터 삽입
    with open(f'{file}.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    for entry in data:
        # title을 Title 테이블에서 참조하는 형태로 수정
        title = session.query(Title).filter_by(title_text=entry['title']).first()
        if not title:
            title = Title(title_text=entry['title'])
            session.add(title)

        story = Story(
            title=title,
            instance_index=entry['instance_index'],
            instance=entry['instance'],
            speaker=entry['speaker'],
            speaker_index=entry['speaker_index'],
            category=entry['category']
        )
        session.add(story)

    # 변경사항 저장 및 세션 닫기
    session.commit()
    session.close()


def db2json(file):
    """
    db 속 story 값을 json 형식으로 저장합니다.
    """
    # SQLAlchemy 세션 생성
    session = Session()

    # 데이터베이스에서 모든 레코드 가져오기
    stories = session.query(Story).all()

    # JSON 파일로 데이터 쓰기
    data = []
    for story in stories:
        data.append({
            'title': story.title.title_text,  # title을 Title 테이블에서 가져오도록 수정
            'instance_index': story.instance_index,
            'instance': story.instance,
            'speaker': story.speaker,
            'speaker_index': story.speaker_index,
            'category': story.category
        })

    with open(f'{file}.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    # 세션 닫기
    session.close()


def show_table_info(table_name='all'):
    # SQLAlchemy Inspector 생성
    inspector = inspect(engine)

    if table_name == 'all':
        tables = inspector.get_table_names()

        # 각 테이블에 대한 정보 출력
        for table_names in tables:
            print(f"\nTable: {table_names}")
            columns = inspector.get_columns(table_names)
            for column in columns:
                print(f"Column: {column['name']}, Type: {column['type']}, "
                      f"Nullable: {column['nullable']}")

    else:
        # 테이블 정보 가져오기
        columns = inspector.get_columns(table_name)

        # 테이블 정보 출력
        print(f"Table: {table_name}")
        for column in columns:
            print(f"Column: {column['name']}, Type: {column['type']}, "
                  f"Nullable: {column['nullable']}")



