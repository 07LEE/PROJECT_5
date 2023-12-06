def get_alias2id(name_list_path) -> dict:
    """
    
    """
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    alias2id = {}
    for i, line in enumerate(name_lines):
        for alias in line.strip().split()[1:]:
            alias2id[alias] = i
    return alias2id


def get_id2alias(name_list_path) -> list:
    """
    
    """
    id2alias = []
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    for line in name_lines:
        id2alias.append(line.strip().split()[1])
    return id2alias
