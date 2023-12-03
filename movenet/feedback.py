import pandas as pd

RANGES = {
    "downdog": [],
    "goddess": [],
    "plank": [],
    "tree": [],
    "warrior": []
}


def writeToFile(data, filepath):
    # df = pd.DataFrame()
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # df = pd.DataFrame()
        new_row = pd.DataFrame([data])
        new_row.to_csv(filepath, index=False, header=False)
    else:
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filepath, index=False, header=False)


def evaluateDowndog(angles, keypoints):
    writeToFile(angles, 'downdog_angles.csv')
    writeToFile(keypoints, 'downdog_keypoints.csv')

    feedback = []

    # L and R elbow to straighten
    if angles[0] < RANGES['downdog'][0][0] and angles[4] < RANGES['downdog'][4][0]:
        feedback.append('Straighten both elbows')
    # left elbow
    elif angles[0] < RANGES['downdog'][0][0]:
        feedback.append('Straighten left elbow')
    # right elbow
    elif angles[4] < RANGES['downdog'][4][0]:
        feedback.append('Straighten right elbow')

    # L and R hip to square (ie 90 degrees ish)
    if angles[2] > RANGES['downdog'][2][1] and angles[6] > RANGES['downdog'][6][1]:
        feedback.append('Square hips')  # raise butt higher?? butt is sagging??

    # L and R knee to straighten
    if angles[3] < RANGES['downdog'][3][0] and angles[7] < RANGES['downdog'][7][0]:
        feedback.append('Straighten both knees')
    # left knee
    elif angles[3] < RANGES['downdog'][3][0] or angles[3] > RANGES['downdog'][3][1]:
        feedback.append('Straighten left knee')
    # right knee
    elif angles[7] < RANGES['downdog'][7][0] or angles[7] > RANGES['downdog'][7][1]:
        feedback.append('Straighten right knee')

    return feedback


def evaluateGoddess(angles, keypoints):
    writeToFile(angles, 'goddess_angles.csv')
    writeToFile(keypoints, 'goddess_keypoints.csv')

    feedback = []

    # L and R wrists above elbows
    if keypoints[9] < keypoints[5] and keypoints[11] < keypoints[7]:
        feedback.append('Raise both wrists above elbows')
    # L wrist above L elbow
    elif keypoints[9] < keypoints[5]:
        feedback.append('Raise left wrist above left elbow')
    # R wrist above R elbow
    elif keypoints[11] < keypoints[7]:
        feedback.append('Raise right wrist above right elbow')

    # L and R elbow need to be bent
    if angles[0] > RANGES['goddess'][0][1] and angles[4] > RANGES['goddess'][4][1]:
        feedback.append('Bend both elbows')
    # L elbow
    elif angles[0] > RANGES['goddess'][0][0]:
        feedback.append('Bend left elbow')
    # R elbow
    elif angles[4] > RANGES['goddess'][4][0]:
        feedback.append('Bend right elbow')

    # L and R shoulder should be square
    if angles[1] < RANGES['goddess'][1][0] and angles[5] > RANGES['goddess'][5][0]:
        feedback.append('Raise both arms')

    # L and R knee should be bent
    if angles[3] > RANGES['goddess'][3][1] and angles[7] > RANGES['goddess'][7][1]:
        feedback.append('Bend both knees more')

    return feedback


def evaluatePlank(angles, keypoints):
    writeToFile(angles, 'plank_angles.csv')
    writeToFile(keypoints, 'plank_keypoints.csv')

    feedback = []

    # L and R elbow
    if angles[0] < RANGES['plank'][0][0] and angles[4] < RANGES['plank'][4][0]:
        feedback.append('Straighten both elbows')
    # left elbow
    elif angles[0] < RANGES['plank'][0][0]:
        feedback.append('Straighten left elbow')
    # right elbow
    elif angles[4] < RANGES['plank'][4][0]:
        feedback.append('Straighten right elbow')

    # L and R shoulder should be square
    if angles[1] > RANGES['plank'][1][1] and angles[5] > RANGES['plank'][5][1]:
        feedback.append('Square shoulders')

    # L and R hip should be square
    if (angles[2] > RANGES['plank'][2][1] and angles[6] > RANGES['plank'][6][1]) or \
            (angles[2] < RANGES['plank'][2][0] and angles[6] < RANGES['plank'][6][0]):
        feedback.append('Keep hips square and level')

    # L and R knee should be straight
    if angles[3] < RANGES['plank'][3][0] and angles[7] < RANGES['plank'][7][0]:
        feedback.append('Straighten both knees')
    # left knee
    elif angles[3] < RANGES['plank'][3][0]:
        feedback.append('Straighten left knee')
    # right knee
    elif angles[7] < RANGES['plank'][7][0]:
        feedback.append('Straighten right knee')

    return feedback


def evaluateTree(angles, keypoints):
    writeToFile(angles, 'tree_angles.csv')
    writeToFile(keypoints, 'tree_keypoints.csv')

    feedback = []

    # L and R wrist above elbows
    if keypoints[9] < keypoints[5] and keypoints[11] < keypoints[7]:
        feedback.append('Raise your arms')

    # L and R elbow should be straight
    if angles[0] < RANGES['tree'][0][0] and angles[4] < RANGES['tree'][4][0]:
        feedback.append('Straighten both elbows')

    # L and R knee
    # one knee should be bent and the other should be straight
    if (angles[3] < RANGES['tree'][3][0] and angles[7] > RANGES['tree'][7][1]) or \
            (angles[3] > RANGES['tree'][3][1] and angles[7] < RANGES['tree'][7][0]):
        feedback.append('Straighten one knee and bend the other')

    return feedback


def evaluateWarrior(angles, keypoints):
    writeToFile(angles, 'warrior_angles.csv')
    writeToFile(keypoints, 'warrior_keypoints.csv')

    feedback = []

    # L and R elbow should be straight
    if angles[0] < RANGES['warrior'][0][0] and angles[4] < RANGES['warrior'][4][0]:
        feedback.append('Straighten both elbows')
    # left elbow
    elif angles[0] < RANGES['warrior'][0][0]:
        feedback.append('Straighten left elbow')
    # right elbow
    elif angles[4] < RANGES['warrior'][4][0]:
        feedback.append('Straighten right elbow')

    # L and R shoulder should be square
    if angles[1] > RANGES['warrior'][1][1] and angles[5] > RANGES['warrior'][5][1]:
        feedback.append('Square shoulders')

    # L and R knee - 1 knee bent, 1 knee straight
    if (angles[3] < RANGES['warrior'][3][0] and angles[7] > RANGES['warrior'][7][1]) or \
            (angles[3] > RANGES['warrior'][3][1] and angles[7] < RANGES['warrior'][7][0]):
        feedback.append('Straighten one knee and bend the other')

    return feedback


FEEDBACK_FUNCS = {
    "downdog": evaluateDowndog,
    "goddess": evaluateGoddess,
    "plank": evaluatePlank,
    "tree": evaluateTree,
    "warrior": evaluateWarrior
}


def setRanges():
    df = pd.read_csv('pose_ranges.csv', header=[0, 1])
    poses = ['downdog', 'goddess', 'plank', 'tree', 'warrior']
    for index, pose in enumerate(poses):
        pose_stats = df.iloc[index].tolist()
        pose_stats = [(pose_stats[i], pose_stats[i + 1]) for i in range(0, len(pose_stats), 2)]
        RANGES[pose] = pose_stats


def preprocess_angles(angles):
    df = pd.DataFrame(angles)
    angles_list = df.values.flatten().tolist()
    return angles_list


def preprocess_keypoints(keypoints):
    keypoint_list = keypoints.tolist()
    modified_keypoints_list = [item for sublist in keypoint_list for index, item in enumerate(sublist) if index != 2]
    return modified_keypoints_list


def evaluatePose(pose, angles, keypoints):
    print(pose)
    angles = preprocess_angles(angles)
    print(angles)
    keypoints = preprocess_keypoints(keypoints)
    print(keypoints)
    setRanges()
    feedback = FEEDBACK_FUNCS[pose](angles, keypoints)
    print(feedback)
    return feedback


def main():
    setRanges()
    downdog_keypoints = [0.5852189064025879, 0.4396080374717712, 0.5901364088058472, 0.4235751330852508,
                         0.5886564254760742, 0.4301187992095947, 0.5728956460952759, 0.3754459023475647,
                         0.5712627172470093, 0.3847272992134094, 0.5037689208984375, 0.3762058615684509,
                         0.4972769021987915, 0.3873338103294372, 0.6012091636657715, 0.325333833694458,
                         0.5888140797615051, 0.3335884213447571, 0.7069939970970154, 0.2683448493480682,
                         0.674348771572113, 0.2884176969528198, 0.3856688737869262, 0.636243462562561,
                         0.3788259029388428, 0.6280312538146973, 0.5469187498092651, 0.7459497451782227,
                         0.531103253364563, 0.7450016736984253, 0.6750020384788513, 0.8729957342147827,
                         0.6404932737350464, 0.8503272533416748, 179.25588004186864, 141.99425691841566,
                         99.80349635854085, 169.46234629358418, 177.41944151476386, 146.6215406734187,
                         101.32680584295007, 173.6137765236629]

    # # remove nose, eyes, ears
    downdog_keypoints = downdog_keypoints[10:34]

    downdog_angles = [179.25588, 141.9942569, 99.80349636, 169.4623463, 177.4194415, 146.6215407, 101.3268058,
                      173.6137765]

    feedback = FEEDBACK_FUNCS["downdog"](downdog_angles, downdog_keypoints)
    print(feedback)


if __name__ == "__main__":
    main()
