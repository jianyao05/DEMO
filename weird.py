
'''
while True:
    x = int(input('hello'))
    list.append(x)
    if list == [1]:
        print('value received')
    else:
        print('mid')
    print(list)
'''
List = []
INCORRECT_POSTURE = False
SQUAT_COUNT = 0
IMPROPER_SQUAT = 0
def get_state(self, angle):
    if 0 <= angle <= 30:
        state = 1
    elif 35 <= angle <= 75:
        state = 2
    elif 80 <= angle <= 100:
        state = 3
    return f"s{state}" if state else None

def update_state_sequence(state):
    if state == 's2':
        if (('s3' not in list) and (list.count('s2')) == 0) or (('s3' in list) and (list.count('s2') == 1)):
            list.append(state)
            '''If 's3' hasn’t been added yet, only one 's2' can be added.
               If 's3' has been added, one more 's2' can be added, but only if it has appeared once before.'''
    elif state == 's3':
        if (state not in list) and ('s2' in list):
            list.append(state)

def idk():
    global list, INCORRECT_POSTURE,SQUAT_COUNT, IMPROPER_SQUAT
    if current_state == 's1':
        if len(list) == 3 and not INCORRECT_POSTURE:
            SQUAT_COUNT += 1

        elif 's2' in list and len(list) == 1:
            IMPROPER_SQUAT += 1


        elif INCORRECT_POSTURE:
            IMPROPER_SQUAT += 1

        list = []
        INCORRECT_POSTURE = False
