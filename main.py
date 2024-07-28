import socket
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from plotter import plot, transform_points, transform_points_inverse, transform_matrix, update_occupancy_grid, update_free_space_grid
from particle_filter import icp
from openCV_display import process_frame

#! Grid ---------------------------

# Constants
FOV_DEGREES = 101
NUM_READINGS = 101
CELL_SIZE = 2
WINDOW_SIZE = 5

# Initialize an empty occupancy grid map
map_width = 107//(CELL_SIZE - 1)
map_height = 107//(CELL_SIZE - 1)
occupancy_grids = [np.zeros((map_height, map_width))]
free_space_grids = [np.zeros((map_height, map_width))]

probability_filter = 0

#ICP
icp_iterations = 2
icp_tolerance = 0.0001

#display
display_scale = 6
guide_color = (225, 86, 43)

#! Self port ----------------------

localIP = socket.gethostbyname(socket.gethostname())
localPort = 1234

#! ESP32 port ----------------------

espIP = None
espPort = 1234 

bufferSize = 2024

print(f"Own Local IP Address: {localIP}")

# Create a datagram socket
serverSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
serverSocket.bind(("", localPort))

# don't block the socket
serverSocket.setblocking(False)

print(f"UDP server up and listening on port {localPort}")

#! Scans ---------------------------

previous_coordinates = []
coordinates = None

robot_pose = (0, 0, 0)  # (x, y, theta)
robot_pose = np.array([
    [np.cos(robot_pose[2]), -np.sin(robot_pose[2]), robot_pose[0]],
    [np.sin(robot_pose[2]),  np.cos(robot_pose[2]), robot_pose[1]],
    [0,              0,             1]
])

# Listen for incoming datagrams
while True:

    #! Receiving data ---------------------------------------------------

    #? Send data to the ESP32 -------------------------------------------

    if espIP is not None:
        #check if "s" is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            to_send = input("Enter the data to send: ")
            # [f,l,r,b] -> [front, back, left, right, stop] -> [0,1,2,3,4] followed by number (int)
            to_send = to_send.split()
            payload = {
                "d": int(to_send[0]), #direction
                "t": int(to_send[1]), #time
                "s": int(to_send[2]), #speed
            }
            payload = json.dumps(payload)
            print(f"Sending data to ESP32: {payload}")
            serverSocket.sendto(payload.encode('utf-8'), (espIP, espPort))
    
    
    #? Receive data from the ESP32 --------------------------------------

    try:
        message, address = serverSocket.recvfrom(bufferSize)
        message = message.decode('utf-8')  # Decode message to string
        print(address)

        if espIP is None:
            espIP = address[0]
            print(f"ESP32 IP Address: {espIP}")

        data = json.loads(message)
        #print in green color
        print("\033[92m", "Data: ", data, "\033[0m")
        
        '''
        #store in a csv file data, 25
        with open('lidar_calibration_data.csv', 'a') as f:
            f.write(f"{data}, {10.4}\n")'''

        #! Calculations ---------------------------------------------------

        #* Calculate (x, y) coordinates of the LiDAR points ---------------
        coordinates = plot(data, 40, 140, 220, 320, NUM_READINGS) # -----> [(x1, y1), (x2, y2), ... ]

        if(previous_coordinates):

            #! ICP ---------------------------------------------------------
            #to numpy array
            A = np.concatenate(previous_coordinates[-WINDOW_SIZE:])
            B = np.array(list(coordinates))
            
            #print in blue color
            print("\033[94m", "Coordinates: ", B, "\033[0m")
            
            #check if B is empty
            if B.size == 0:
                print("Input arrays A and B must not be empty.")
                continue

            # Perform ICP
            T_final, distances, iterations = icp(A, B, max_iterations=icp_iterations, tolerance=icp_tolerance)
            #inverse of T
            T_inv = np.linalg.inv(T_final)

            #New coordinates
            B = transform_points_inverse(T_final, B)
            coordinates = B

            #new robot pose
            robot_pose = T_inv
        
        # Update past coordinates
        previous_coordinates.append(np.array(list(coordinates)))
        if len(previous_coordinates) > WINDOW_SIZE:
            previous_coordinates.pop(0)

        #* Update the occupancy grid map ----------------------------------

        occupancy_grid = update_occupancy_grid(coordinates, map_height, map_width, CELL_SIZE, occupancy_grids, alpha=0.9)

        # Update past occupancy grids
        occupancy_grids.append(occupancy_grid)
        if(len(occupancy_grids) > WINDOW_SIZE):
            occupancy_grids.pop(0)

        free_space_grid, robot_cell, robot_rotation = update_free_space_grid(robot_pose, coordinates, map_height, map_width, CELL_SIZE, free_space_grids, alpha=0.9)

        # Update past free space grids
        free_space_grids.append(free_space_grid)
        if(len(free_space_grids) > WINDOW_SIZE):
            free_space_grids.pop(0)

        #filter out cells with low probability
        occupancy_grid[occupancy_grid < probability_filter] = 0
        free_space_grid[free_space_grid < probability_filter] = 0

        #! Display ---------------------------------------------------------------

        processed_occupancy_grid = process_frame(occupancy_grid, cv2.COLORMAP_JET, robot_cell, robot_rotation, display_scale, guide_color)
        processed_free_space_grid = process_frame(free_space_grid, cv2.COLOR_GRAY2BGR, robot_cell, robot_rotation, display_scale, guide_color)
        
        # Display the occupancy grid map
        cv2.imshow('Occupancy Grid Map', processed_occupancy_grid)
        cv2.imshow('Free Space Grid Map', processed_free_space_grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            cv2.destroyAllWindows()  # Close all OpenCV windows
            break  # Exit the loop or program

    except BlockingIOError:

        # No data is available, skip
        pass