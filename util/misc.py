from datetime import datetime


def get_cur_time():
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

def parse_original_question(question):
    return question.split('\n')[0]

def merge_question(question_new, question):
    return question_new + '\n' + '\n'.join(question.split('\n')[1:])