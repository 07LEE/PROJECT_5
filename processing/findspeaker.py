"""
Author: 
"""
def print_not_instance(self, i: int):
    print(self.text[i]['title'])
    print(self.text[i]['instance_index'])
    print(self.text[i]['speaker'])
    print(self.text[i]['speaker_index'])
    print(self.text[i]['category'])


def print_instance(self, i: int):
    global title_text
    print(self.text[i]['instance'])
    title_text = self.text[i]['title']
