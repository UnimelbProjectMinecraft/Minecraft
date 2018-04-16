from __future__ import print_function

import json
import logging
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

class Qlearning(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.01  # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        # q-table after 300 times train
        self.q_table = {'0:1': [-202.0, -201.0, -127.0, -28.0], '0:0': [-200.0, -100.0, -200.0, -200.0],
                        '1:1': [-101.0, -101.0, -28.0, -27.0], '1:0': [-100.0, -100.0, -200.0, -100.0],
                        '2:1': [-101.0, -101.0, -28.0, -26.0], '3:1': [-102.0, -101.0, -26.0, -25.0],
                        '1:2': [-100.0, -200.0, -100.0, -200.0], '0:2': [-200.0, -300.0, -200.0, -200.0],
                        '2:2': [-100.0, -100.0, -200.0, -200.0], '2:0': [-100.0, -100.0, -201.0, -200.0],
                        '4:1': [-101.0, -101.0, -26.0, -25.0], '3:2': [-200.0, -200.0, -200.0, -200.0],
                        '3:0': [-100.0, -100.0, -100.0, -100.0], '4:0': [-200.0, -100.0, -100.0, -100.0],
                        '4:2': [-200.0, -200.0, -200.0, -300.0], '5:1': [-102.0, -24.0, -201.0, -101.0],
                        '6:1': [-100.0, -200.0, -100.0, -101.0], '5:0': [-100.0, -101.0, 0, 0],
                        '5:2': [-25.0, -23.0, -101.0, -101.0], '6:2': [-200.0, -200.0, -101.0, -200.0],
                        '5:3': [-23.0, -102.0, -22.0, -202.0], '6:3': [-101.0, -100.0, -101.0, -200.0],
                        '4:3': [-202.0, -101.0, -21.0, -22.0], '3:3': [-101.0, -20.0, -101.0, -22.0],
                        '5:4': [-201.0, -200.0, -200.0, -200.0], '4:4': [-200.0, -200.0, -200.0, -200.0],
                        '2:3': [-200.0, -100.0, -100.0, -100.0], '3:4': [-20.0, -19.0, -101.0, -101.0],
                        '2:4': [-100.0, -100.0, -100.0, -200.0], '3:5': [-18.0, -101.0, -102.0, -18.0],
                        '4:5': [-101.0, -102.0, -19.0, -18.0], '4:6': [-100.0, -200.0, -100.0, -100.0],
                        '3:6': [-100.0, -100.0, -100.0, -200.0], '2:5': [-100.0, -200.0, 0, -100.0],
                        '5:5': [-202.0, -17.0, -17.0, -101.0], '5:6': [-17.0, -16.0, -101.0, -101.0],
                        '6:6': [-200.0, -100.0, -200.0, -200.0], '6:5': [-200.0, -200.0, -200.0, -201.0],
                        '5:7': [-16.0, -101.0, -102.0, -15.0], '6:7': [-202.0, -101.0, -16.0, -14.0],
                        '5:8': [-100.0, -100.0, -200.0, -200.0], '4:7': [-100.0, -100.0, -100.0, 0],
                        '7:7': [-13.0, -101.0, -14.0, -101.0], '6:8': [-201.0, -201.0, -101.0, -200.0],
                        '7:8': [-100.0, -200.0, -100.0, -101.0], '8:7': [-200.0, -100.0, -100.0, -200.0],
                        '7:6': [-12.0, -14.0, -101.0, -101.0], '8:6': [-200.0, -100.0, -200.0, -100.0],
                        '7:5': [-101.0, -113.0, -102.0, -12.0], '8:5': [-101.0, -102.0, -11.0, -10.0],
                        '9:5': [-10.0, -9.0, -10.0, -102.0], '9:6': [-10.0, -9.0, -101.0, -101.0],
                        '8:4': [-200.0, -101.0, -200.0, -101.0], '9:4': [-10.0, -9.0, -101.0, -102.0],
                        '9:3': [-102.0, -10.0, -11.0, -203.0], '8:3': [-102.0, -101.0, -10.0, -11.0],
                        '10:5': [0, 0, 0, -100.0], '9:7': [-8.0, -8.0, -101.0, -102.0],
                        '9:8': [-8.0, -7.0, -101.0, -101.0], '8:8': [-200.0, -200.0, -100.0, -101.0],
                        '10:4': [0, -100.0, 0, -100.0], '10:6': [-101.0, -200.0, -100.0, -100.0],
                        '10:3': [0, 0, 0, -101.0], '10:7': [0, 0, 0, -100.0], '9:2': [-100.0, -100.0, 0, -100.0],
                        '8:2': [-201.0, -101.0, -200.0, -100.0], '7:3': [-10.0, -101.0, -101.0, -10.0],
                        '7:2': [-9.0, -11.0, -101.0, -101.0], '9:9': [-8.0, -102.0, -6.0, -102.0],
                        '8:9': [-101.0, -101.0, -5.0, -7.0], '10:8': [-200.0, -100.0, -100.0, -100.0],
                        '7:4': [-201.0, -100.0, -100.0, -101.0], '9:10': [0, -100.0, 0, 0],
                        '10:9': [-100.0, 0, 0, -100.0], '7:1': [-101.0, -10.0, -101.0, -11.0],
                        '7:0': [-200.0, -201.0, -100.0, -100.0], '8:10': [-200.0, -100.0, -200.0, -100.0],
                        '7:9': [-102.0, -102.0, -4.0, -6.0], '6:9': [-101.0, -101.0, -3.0, -5.0],
                        '7:10': [0, 0, 0, -100.0], '8:1': [-102.0, -101.0, -10.0, -12.0], '8:0': [0, -100.0, 0, 0],
                        '9:1': [-102.0, -101.0, -11.0, -102.0], '6:10': [-200.0, -100.0, -101.0, -100.0],
                        '5:9': [-101.0, -101.0, -2.0, -4.0], '4:9': [-1.0, -101.0, -1.0, -2.0],
                        '3:9': [-1.0, -1.0, -1.0, -2.0], '3:10': [0, 0, 0, -100.0], '9:0': [0, -101.0, 0, -100.0],
                        '5:10': [-100.0, -100.0, -100.0, -200.0], '10:1': [0, 0, 0, -100.0],
                        '4:8': [-100.0, -100.0, -100.0, 0], '4:10': [-200.0, -100.0, -100.0, -100.0],
                        '2:9': [0, -1.0, 0, 0], '2:10': [0, 0, 0, -100.0], '3:8': [0, 0, 0, -100.0]}
        self.canvas = None
        self.root = None

    def updateQTable(self, reward, current_state):
        """Change q_table to reflect what we have learnt."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        # old_q = self.q_table[self.prev_s][self.prev_a]
        maxvalue = max(self.q_table[current_state])
        new_q = maxvalue + reward
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def updateQTableFromTerminatingState(self, reward):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        new_q = reward + self.q_table[self.prev_s][self.prev_a]
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def act(self, world_state, agent_host, current_r):
        """take 1 action in response to the current world state"""

        obs_text = world_state.observations[-1].text
        obs = json.loads(obs_text)  # most recent observation
        self.logger.debug(obs)
        if not u'XPos' in obs or not u'ZPos' in obs:
            self.logger.error("Incomplete observation received: %s" % obs_text)
            return 0
        current_s = "%d:%d" % (int(obs[u'XPos']), int(obs[u'ZPos']))
        self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if current_s not in self.q_table:
            self.q_table[current_s] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable(current_r, current_s)

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][x] == m:
                    l.append(x)
            y = random.randint(0, len(l) - 1)
            a = l[y]
            self.logger.info("Taking q action: %s" % self.actions[a])

        # try to send the selected action, only update prev_s if this succeeds
        try:
            agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0

        self.prev_s = None
        self.prev_a = None

        is_first_action = True

        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:

            current_r = 0

            if is_first_action:
                # wait until have received a valid observation
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.1)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.observations) > 0 and not \
                            world_state.observations[-1].text == "{}":
                        total_reward += self.act(world_state, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState(current_r)

        return total_reward


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
                  <ServerQuitFromTimeUp timeLimitMs="1000000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Miki</Name>
                <AgentStart>
                  <Placement x="0.5" y="228.0" z="1.5" pitch="23.5" yaw="-90"/>
                  <Inventory>
                    <InventoryItem slot="20" type="diamond_pickaxe"/>
                  </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <DiscreteMovementCommands/>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                  <RewardForTouchingBlockType>
                    <Block reward="100.0" type="gold_block" behaviour="onceOnly"/>
                    <Block reward="-100.0" type="lava" behaviour="onceOnly"/>
                  </RewardForTouchingBlockType>
                  <RewardForSendingCommand reward="-1" />
                  <AgentQuitFromTouchingBlockType>
                      <Block type="lava" />
                      <Block type="gold_block" />
                  </AgentQuitFromTouchingBlockType>

                </AgentHandlers>
              </AgentSection>
            </Mission>'''


# draw the un-separated maze
def DrawMazeBase(my_mission, length, width, block_type, hole_type):
    # to draw a maze base that each block considered as a cell
    length_m = 2 * length
    width_m = 2 * width
    my_mission.drawCuboid(-20, 225, -20, length_m + 20, 226, width_m + 20, hole_type)
    # my_mission.drawCuboid(0, 226, 0, length_m, 226, width_m, 'stone')

    for i in range(length):
        for j in range(width):
            leng = 2 * i + 1
            wid = 2 * j + 1
            my_mission.drawCuboid(leng, 226, wid, leng, 227, wid, block_type)


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
    my_mission.drawCuboid(0, 226, 1, 0, 227, 1, "diamond_block")
    # my_mission.drawBlock(0, 226, 1, "diamond_block")
    # the goal state
    my_mission.drawCuboid(x_goal, 226, z_goal, x_goal, 227, z_goal, "gold_block")

    # my_mission.drawBlock(x_goal, 226, z_goal + 1, "gold_block")

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
            my_mission.drawCuboid(x_n, 226, z_n, x_p, 227, z_p, "stone")


# put resource randomly in the maze
def release_resource(item_no, length, width):
    blcok_list = ['diamond_block', 'stone', 'wool', 'glod_ore', 'diamond_ore', 'iron_ore',
                  'dragon_egg', 'carpet', 'hopper', 'carrots', 'beacon', 'cocoa', 'cake', 'reeds', 'lever',
                  'wheat', 'piston', 'bed']
    item_list = random.sample(blcok_list, item_no)
    for i in range(item_no):
        length_random = random.randint(1, length - 1) * 2 + 1
        width_random = random.randint(1, width - 1) * 2 + 1
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

# read xml from outside file
mission_file = './lava_maze.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

# draw a maze that each cell is separate from each other
DrawMazeBase(my_mission, length, width, "stonebrick", "lava")
WalkMaze(my_mission, length, width)
release_resource(resource_no, length, width)

# implement qlearning
agent = Qlearning()

# Attempt to start a mission
max_retries = 3
cumulative_rewards = []
total_repeat = 300
# start learning
for i in range(total_repeat):
    print()
    print('Repeat %d of %d' % (i + 1, total_repeat))
    my_mission_record = MalmoPython.MissionRecordSpec()

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

    cumulative_reward = agent.run(agent_host)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [cumulative_reward]
    time.sleep(0.5)

print('Q-table: ', agent.q_table)
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
