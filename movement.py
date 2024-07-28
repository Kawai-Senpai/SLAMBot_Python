import numpy as np

def forward(distance):

    '''
    Input: distance (float) - distance from the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()
    #return a * x**3 + b * x**2 + c * x + d
    time = weights['forward'][0] * distance ** 3 + weights['forward'][1] * distance ** 2 + weights['forward'][2] * distance + weights['forward'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['forward'])

    return {
        "d": 0,
        "t": time,
        "s": 100
    }

def backward(distance):

    '''
    Input: distance (float) - distance from the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()
    
    time = weights['backward'][0] * distance ** 3 + weights['backward'][1] * distance ** 2 + weights['backward'][2] * distance + weights['backward'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['backward'])

    return {
        "d": 1,
        "t": time,
        "s": 100
    }

def left(degree):

    '''
    Input: distance (float) - degree wtr the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()

    time = weights['left'][0] * degree ** 3 + weights['left'][1] * degree ** 2 + weights['left'][2] * degree + weights['left'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['left'])

    return {
        "d": 2,
        "t": time,
        "s": 200
    }

def right(degree):

    '''
    Input: distance (float) - degree wtr the origin
    Output: Time (float) - time to tuen on motors (payload)
    '''

    weights = np.load('movement_weights.npy', allow_pickle=True).item()

    time = weights['right'][0] * degree ** 3 + weights['right'][1] * degree ** 2 + weights['right'][2] * degree + weights['right'][3]
    #round to integer
    time = round(time)
    #print("weights: ", weights['right'])

    return {
        "d": 3,
        "t": time,
        "s": 200
    }
