from __future__ import print_function

import json
import logging
import os
import random
import sys
import time
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
        self.q_table = {}
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



# draw the un-separated maze
def DrawMazeBase(my_mission, length, width, block_type):
    # to draw a maze base that each block considered as a cell
    length_m = 2 * length
    width_m = 2 * width

    my_mission.drawCuboid(0, 226, 0, 0, 230, length_m, block_type)
    my_mission.drawCuboid(0, 226, 0, width_m, 230, 0, block_type)
    my_mission.drawCuboid(width_m, 226, 0, width_m, 230, length_m, block_type)
    my_mission.drawCuboid(width_m, 226, length_m, 0, 230, length_m, block_type)

    # x = random.randint(1, width_m)
    # z = random.randint(1, length_m)
    my_mission.drawBlock(15, 226, 16, 'gold_block')


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



if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
else:
    import functools
    print = functools.partial(print, flush=True)



# implement qlearning
agent = Qlearning()
#  start minecraft
agent_host = MalmoPython.AgentHost()
agent_host.addOptionalStringArgument("size", "The size of the maze.", "10*10")
agent_host.addOptionalIntArgument("res", "The number of resource want to add", 4)
# sys.argv receive the command arguments
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host.getUsage())
    exit(1)

if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)
# get the size of maze from command line
size_of_maze = agent_host.getStringArgument("size")
size_list = size_of_maze.split('*', 1)
resource_no = agent_host.getIntArgument("res")
agent_host.getIntArgument("res")
length = int(size_list[0])
width = int(size_list[1])

# read xml from outside file
mission_file = './flat_maze.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)

# draw a maze that each cell is separate from each other
DrawMazeBase(my_mission, length, width, "stonebrick")
release_resource(resource_no, length, width)

# Attempt to start a mission
max_retries = 3
cumulative_rewards = []
total_repeat = 3000
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

    # agent_host.sendCommand('move 1')
    # print("move to north")
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

# print('Q-table: ', agent.q_table)
# print("Mission running ", end=' ')

# agent_host.sendCommand("move 1")
#
# while world_state.is_mission_running:
#     print(".", end="")
#     time.sleep(0.1)
#     world_state = agent_host.getWorldState()
#     for error in world_state.errors:
#         print("Error:", error.text)

print()
print("Mission ended")
# Mission has ended.