from __future__ import print_function

import os
import random
import sys
import time
from Stack import Stack
from builtins import range

import MalmoPython

# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
else:
    import functools

    print = functools.partial(print, flush=True)


# create the map environment
def GetMissionXML():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
            xmlns:schemaLocation="http://ProjectMalmo.microsoft.com Mission.xsd">
            
              <About>
                <Summary>Come on maze!</Summary>
              </About>
              
              <ServerSection>
                <ServerInitialConditions>
                  <Time>
                    <StartTime>10000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                  </Time>
                  <Weather>clear</Weather>
                </ServerInitialConditions>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                  <ServerQuitFromTimeUp timeLimitMs="10000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Miki</Name>
                <AgentStart>
                  <Placement x="1" y="227.0" z="1.5" pitch="23.5" yaw="-83"/>
                  <Inventory>
                    <InventoryItem slot="20" type="diamond_pickaxe"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <DiscreteMovementCommands/>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''


# draw the un-separated maze
def DrawMazeBase(my_mission, length, width, block_type, hole_type):
    # to draw a maze base that each block considered as a cell
    length_m = 2 * length
    width_m = 2 * width
    my_mission.drawCuboid(0, 226, 0, length_m, 228, width_m, block_type)
    for i in range(length):
        for j in range(width):
            leng = 2 * i + 1
            wid = 2 * j + 1
            my_mission.drawCuboid(leng, 227, wid, leng, 228, wid, hole_type)


# separate the maze by bfs
def WalkMaze(my_mission, length, width):
    # use dft to break the walls between cells
    length_m = 2 * length
    width_m = 2 * width
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = []
    parent = None
    start_state = (1, 1)
    x_goal = random.randint(1, length) * 2
    if x_goal == length_m:
        z_goal = random.randint(1, width) * 2 - 1
    else:
        z_goal = width_m - 1
    goal_state = (x_goal, z_goal)
    current = {'state': start_state, 'parent': parent}

    # draw the start state
    my_mission.drawCuboid(0, 227, 1, 0, 228, 1, "air")
    my_mission.drawBlock(0, 226, 1, "diamond_block")
    # the goal state
    my_mission.drawCuboid(x_goal, 227, z_goal, x_goal, 228, z_goal+1, "air")
    my_mission.drawBlock(x_goal, 226, z_goal + 1, "gold_block")

    # inner class, check if agent inside the maze
    def InsideMaze(x_num, z_num):
        if x_num < 1 or x_num > length_m or z_num < 1 or z_num > width_m:
            return False
        else:
            return True

    # inner class, check if the state is goal
    def isGoal(state):
        if state == goal_state:
            return True
        else:
            return False

    # start walk the maze
    stack = Stack()
    stack.push(current)

    while not stack.isEmpty():
        node = stack.pop()
        if isGoal(node):
            break;
        x_n, z_n = node['state']

        if node['state'] in visited:
            continue;
        visited.append(node['state'])
        neighbour_list = [(x_n + d_x * 2, z_n + d_z * 2) for (d_x, d_z) in actions]

        while not len(neighbour_list) <= 0:
            neighbour = random.choice(neighbour_list)
            if neighbour not in visited:
                x, z = neighbour
                if InsideMaze(x, z):
                    # print('neighbour:',neighbour)
                    next_state = {'state': neighbour, 'parent': node}
                    stack.push(next_state)
                    neighbour_list.remove(neighbour)
                # print('Stack push:',next_state['state'])
                else:
                    neighbour_list.remove(neighbour)
            else:
                neighbour_list.remove(neighbour)
        if node['parent'] is not None:
            x_p, z_p = node['parent']['state']
            my_mission.drawCuboid(x_n, 227, z_n, x_p, 228, z_p, "air")


# put resource randomly in the maze
def release_resource(item_no, length, width):
    blcok_list = ['diamond_block', 'stone', 'wool', 'glod_ore', 'diamond_ore', 'iron_ore',
                  'dragon_egg', 'carpet', 'hopper', 'carrots', 'beacon', 'cocoa', 'cake', 'reeds', 'lever',
                  'wheat', 'piston', 'bed']
    item_list = random.sample(blcok_list, item_no)
    for i in range(item_no):
        length_random = random.randint(1, length - 1) * 2
        width_random = random.randint(1, width - 1) * 2
        my_mission.drawItem(length_random, 227, width_random, item_list[i])


#  start minecraft
agent_host = MalmoPython.AgentHost()
agent_host.addOptionalStringArgument("size", "The size of the maze.", "5*5")
agent_host.addOptionalIntArgument("res", "The number of resource want to add", 4)
# sys.argv receive the command arguments
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host.getUsage())
    exit(1)

if agent_host.receivedArgument("help"):
    print(agent_host.getUsage)
    exit(0)
# get the size of maze from command line
size_of_maze = agent_host.getStringArgument("size")
size_list = size_of_maze.split('*', 1)
resource_no = agent_host.getIntArgument("res")
agent_host.getIntArgument("res")
length = int(size_list[0])
width = int(size_list[1])
my_mission = MalmoPython.MissionSpec(GetMissionXML(), True)
my_mission_record = MalmoPython.MissionRecordSpec()
# draw a maze that each cell is separate from each other
DrawMazeBase(my_mission, length, width, "stonebrick", "air")
WalkMaze(my_mission, length, width)
release_resource(resource_no, length, width)

# Attempt to start a mission
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:", e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()

while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:", error.text)

print()
print("Mission running ", end=' ')

# agent_host.sendCommand("move 1")

while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:", error.text)

print()
print("Mission ended")
# Mission has ended.
