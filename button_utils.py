import time

import rtde_io
import rtde_receive

#Input ids
BUTTON_DOWN_ID = 3
GRIPPER_CLOSED_ID = 4

ROBOT_IP = "10.30.21.100"

rtde_io_ = rtde_io.RTDEIOInterface(ROBOT_IP)
rtde_receive_ = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

def wait_for_button():
    while not rtde_receive_.getDigitalInState(BUTTON_DOWN_ID):
        time.sleep(0.1)
    while rtde_receive_.getDigitalInState(BUTTON_DOWN_ID):
        time.sleep(0.1)
    return True
    
def wait_for_gripper(state=False):
    while not (rtde_receive_.getDigitalInState(GRIPPER_CLOSED_ID) == state):
        time.sleep(0.1)
    return True
    
if __name__ == '__main__':
    n = 0
    print(dir(rtde_io_))
    '''while True:
        #wait_for_button()
        print("Button pressed: "+str(n))
        n+=1'''
    
   
