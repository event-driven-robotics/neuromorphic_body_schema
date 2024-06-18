'''
Here we map the sensors from ID to location.
bodyParts: UNKNOWN_BODY_PART, HEAD, TORSO, LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG
skinPart: UNKNOWN_SKIN_PART, TORSO, HAND, FOREARM, UPPER_ARM, FOOT, UPPER_LEG
patchID: 0-N
taxelID: 0-N
'''

# from collections import namedtuple

# skinEvent = namedtuple("skinEvent", ["bodyPart", "skinPart", "patchID", "taxelID"])
# taxel = skinEvent("TORSO", "TORSO", 0, 2)

# from dataclasses import astuple, dataclass

# @dataclass
# class Person:
#     bodyPart: str
#     skinPart: str
#     patchID: int
#     taxelID: int
#     value: float
#     def __iter__(self):
#         return iter(astuple(self))
    
# init
taxelDict = {} 
bodyParts = ["UNKNOWN_BODY_PART", "HEAD", "TORSO", "LEFT_ARM", "RIGHT_ARM", "LEFT_LEG", "RIGHT_LEG"]
skinParts = ["UNKNOWN_SKIN_PART", "TORSO", "HAND", "FOREARM", "UPPER_ARM", "FOOT", "UPPER_LEG"]

# torso
for patchID in range(5):
    # example taxel numbers
    for taxelID in range(12):
        taxelDict[("TORSO", "TORSO", patchID, taxelID)] = 0.0

# left arm (each fingertip is concidered a patch and all palm is a single patch)
# add fingertips
for patchID in range(5):
    for taxelID in range(12):
        taxelDict[("LEFT_ARM", "HAND", patchID, taxelID)] = 0.0
# add palm
for taxelID in range(48):
    taxelDict[("LEFT_ARM", "HAND", 5, taxelID)] = 0.0
# add forearm
for patchID in range(10):
    for taxelID in range(12):
        taxelDict[("LEFT_ARM", "FOREARM", patchID, taxelID)] = 0.0
# add forearm
for patchID in range(12):
    for taxelID in range(12):
        taxelDict[("LEFT_ARM", "UPPER_ARM", patchID, taxelID)] = 0.0

# right arm
# add fingertips
for patchID in range(5):
    for taxelID in range(12):
        taxelDict[("RIGHT_ARM", "HAND", patchID, taxelID)] = 0.0
# add palm
for taxelID in range(48):
    taxelDict[("RIGHT_ARM", "HAND", 5, taxelID)] = 0.0
# add forearm
for patchID in range(10):
    for taxelID in range(12):
        taxelDict[("RIGHT_ARM", "FOREARM", patchID, taxelID)] = 0.0
# add forearm
for patchID in range(12):
    for taxelID in range(12):
        taxelDict[("RIGHT_ARM", "UPPER_ARM", patchID, taxelID)] = 0.0

# add left leg
for taxelID in range(4):
    taxelDict[("LEFT_LEG", "FOOT", 0, taxelID)] = 0.0
# add forearm
for patchID in range(24):
    for taxelID in range(12):
        taxelDict[("LEFT_LEG", "UPPER_LEG", patchID, taxelID)] = 0.0

# add right leg
for taxelID in range(4):
    taxelDict[("RIGHT_LEG", "FOOT", 0, taxelID)] = 0.0
# add forearm
for patchID in range(24):
    for taxelID in range(12):
        taxelDict[("RIGHT_LEG", "UPPER_LEG", patchID, taxelID)] = 0.0
print('init done')