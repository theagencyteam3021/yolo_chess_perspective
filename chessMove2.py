# chessMove.py
#
# Expected input: startPosition, endPosition, occupied
#   startPosition = [A-H][1-8]
#   endPosition = [A-H][1-8]
#   occupied = [True|False] (Is the endPosition occupied?)
#
# Example: A2, C3, False


# Standard Modules
import sys
import math
from time import sleep

# Installed Modules
import rtde_control
import rtde_receive
import rtde_io

# Custom Modules
import board # Position Values

# Variables
#==============================================================================
#robotIP = "172.20.92.238"
robotIP = "10.30.21.100"

dropdist = 0.077 # meters of drop and pick


# Initialization
#==============================================================================
rtde_c = rtde_control.RTDEControlInterface(robotIP)
rtde_r = rtde_receive.RTDEReceiveInterface(robotIP)
rtde_io = rtde_io.RTDEIOInterface(robotIP)

rtde_c.setPayload(1.330,[.009,.007,.039])

rtde_io.setStandardDigitalOut(5, False)



# Functions
#==============================================================================
def JointRad(lstDegree):
    lstRad = [math.radians(i)for i in lstDegree]
    return lstRad


def pick(position, board=board):
    print ("Moving to {}".format(position))
    lstJoints = getattr(board, position)
    rtde_c.moveJ(JointRad(lstJoints), 0.5, 0.3)
    target = rtde_r.getActualTCPPose()
    target[2] -= dropdist
    rtde_c.moveL(target) #drop
    sleep(1)
    rtde_io.setStandardDigitalOut(5, True)
    sleep(1)
    target[2] += dropdist
    rtde_c.moveL(target) #pick

def drop(position, board=board):
    print ("Moving to {}".format(position))
    lstJoints = getattr(board, position)
    rtde_c.moveJ(JointRad(lstJoints), 0.5, 0.3)
    target = rtde_r.getActualTCPPose()
    target[2] -= dropdist
    rtde_c.moveL(target) #drop
    sleep(1)
    rtde_io.setStandardDigitalOut(5, False)
    sleep(1)
    target[2] += dropdist
    rtde_c.moveL(target) #pick


def drop_capture():
    lstJoints = getattr(board, 'CAPTURE') # Temporarily using H8 as discard until location is defined
    rtde_c.moveJ(JointRad(lstJoints), 0.5, 0.3)
    rtde_io.setStandardDigitalOut(5, False)
    ### Open Grip here ... TBD





def move(startPosition, endPosition, occupied=True):
    print('Moving from {} to {}'.format(startPosition, endPosition))
    if occupied:
        pick(endPosition)
        drop_capture()
    pick(startPosition)
    drop(endPosition)



# Interactive from Command Line
#==============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Expecting 3 arguements')
    else:
        startPosition = sys.argv[1].upper()
        endPosition = sys.argv[2].upper()
        occupied = sys.argv[3]

        move(startPosition, endPosition, occupied)

        # print('{} {} {}'.format(startPosition, endPosition, occupied))

    
