import pandas as pd
import  numpy as np

def get_ex_data(path):
    ex = pd.read_csv(path)
    # 0 - anger
    # 3 - smile
    # 4 - sad
    # 5 - surprise
    # 6 - neutral
    ex = ex[ex.emotion.isin([0, 3, 4, 5, 6])]

    reverseKey = dict()
    reverseKey['0'] = 'anger'
    reverseKey['3'] = 'smile'
    reverseKey['4'] = 'sad'
    reverseKey['5'] = 'surprise'
    reverseKey['6'] = 'neutral'
    key = dict()
    key['neutral'] = np.array([1, 0, 0, 0, 0])
    key['anger'] = np.array([0, 1, 0, 0, 0])
    key['surprise'] = np.array([0, 0, 1, 0, 0])
    key['smile'] = np.array([0, 0, 0, 1, 0])
    key['sad'] = np.array([0, 0, 0, 0, 1])

    ex.emotion = [reverseKey[str(i)] for i in ex.emotion]
    ex_img = []
    ex_y = []
    for j in range(len(ex)):
        ex_img.append(np.array([int(i) for i in list(ex.pixels)[j].split()]).reshape((48, 48)))
        ex_y.append(key[list(ex.emotion)[j]])

    ex_img = np.array(ex_img)
    ex_y = np.array(ex_y)

    return ex_img, ex_y