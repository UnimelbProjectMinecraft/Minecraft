from __future__ import print_function

import json
import logging
import os
import random
import numpy as np
import sys
import time
from builtins import range

import MalmoPython

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk


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
        self.epsilon = 0.3  # chance of taking a random action instead of the best

        self.logger = logging.getLogger(__name__)
        # disable logger print out
        self.logger.disabled = True
        if False:  # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.q_table ={}
            # {'4:1': [40680.0, 44542.0, 40681.0, 33749.0], '3:1': [33058.0, 43949.0, 36720.0, 37116.0], '3:2': [41077.0, 51967.0, 38307.0, 41275.0], '4:2': [40680.0, 44541.0, 47315.0, 43353.0], '5:2': [54143.0, 55927.0, 46621.0, 54143.0], '5:3': [55530.0, 56126.0, 54540.0, 55035.0], '5:4': [55729.0, 55729.0, 56127.0, 53054.0], '5:5': [55730.0, 54540.0, 42763.0, -3.0], '5:6': [54541.0, 45338.0, 49989.0, -3.0], '6:5': [53054.0, 39000.0, -4.0, 40186.0], '6:4': [55530.0, 52457.0, 47216.0, 51665.0], '4:4': [54837.0, 54344.0, 56128.0, 55829.0], '4:3': [46621.0, 28407.0, 48207.0, 56026.0], '3:3': [45830.0, 55831.0, 41773.0, 55233.0], '6:3': [55529.0, 55430.0, 55828.0, 54440.0], '7:3': [55528.0, 49390.0, 54441.0, 54142.0], '7:2': [9789.0, 55230.0, 55727.0, 55230.0], '7:1': [-8.0, 55231.0, 41569.0, -8.0], '6:2': [52756.0, 55728.0, 55827.0, 50776.0], '8:3': [51664.0, 50181.0, 55230.0, 51862.0], '9:3': [48992.0, 43547.0, 52457.0, 49584.0], '9:4': [-8.0, -8.0, 51369.0, -9.0], '9:5': [-7.0, -7.0, -7.0, -8.0], '9:6': [-8.0, -6.0, 1378.0, -6.0], '9:7': [-6.0, -7.0, 686.0, -7.0], '10:6': [-6.0, -6.0, -7.0, -7.0], '10:5': [-9.0, -7.0, -7.0, -7.0], '10:4': [37506.0, -8.0, -8.0, -9.0], '10:3': [55228.0, -9.0, 48003.0, 49584.0], '10:2': [53940.0, 52453.0, 55526.0, 55228.0], '10:7': [-6.0, -7.0, -6.0, -7.0], '8:7': [586.0, -5.0, 40188.0, -5.0], '7:7': [40187.0, -3.0, 44941.0, 1775.0], '6:7': [-3.0, 24447.0, 45338.0, 33753.0], '6:8': [24448.0, -3.0, -3.0, -4.0], '7:8': [-4.0, -5.0, 93.0, -5.0], '8:8': [686.0, -5.0, -5.0, -5.0], '8:9': [-6.0, 35234.0, -5.0, 7316.0], '9:8': [-6.0, -6.0, -6.0, -7.0], '9:9': [-6.0, -7.0, 35233.0, -8.0], '10:9': [-7.0, -8.0, 6425.0, -8.0], '10:10': [4444.0, 4346.0, 35134.0, 6326.0], '9:10': [7316.0, 7119.0, 35234.0, 6326.0], '8:10': [4050.0, 6427.0, 35235.0, 7119.0], '10:8': [-7.0, -8.0, -7.0, -7.0], '7:10': [34246.0, 35235.0, 38602.0, 4348.0], '7:9': [-5.0, -5.0, 35235.0, -5.0], '6:10': [1282.0, 38602.0, 38801.0, -4.0], '6:9': [786.0, 38503.0, 34248.0, -5.0], '5:9': [-3.0, -3.0, 38308.0, -3.0], '5:10': [-3.0, 6331.0, 38802.0, 4449.0], '5:8': [-3.0, -3.0, 6631.0, -3.0], '4:9': [-2.0, 7421.0, 39695.0, -2.0], '4:8': [-1.0, -2.0, 39795.0, -3.0], '3:8': [44944.0, 39695.0, 7920.0, 34646.0], '4:7': [-2.0, 39200.0, 52270.0, -3.0], '5:7': [44644.0, -3.0, 50586.0, 41377.0], '8:6': [-6.0, -5.0, 34148.0, -5.0], '7:6': [-5.0, 44643.0, -4.0, 34147.0], '6:6': [38999.0, 40486.0, -4.0, -4.0], '4:6': [55037.0, -2.0, -2.0, 44937.0], '7:5': [51271.0, 40187.0, 38999.0, -5.0], '8:5': [33749.0, 33751.0, -6.0, -7.0], '9:2': [51660.0, 51862.0, 55527.0, 55228.0], '8:2': [53150.0, 55229.0, 55627.0, 55229.0], '8:4': [52457.0, 33750.0, 44937.0, 33748.0], '7:4': [53648.0, 40186.0, 51666.0, 44936.0], '5:1': [52757.0, 55431.0, -7.0, 6818.0], '6:1': [-8.0, -7.0, 52757.0, 41568.0], '2:2': [35928.0, 40684.0, 29693.0, 32168.0], '8:1': [-8.0, 53844.0, -8.0, 51860.0], '9:1': [38295.0, 53645.0, 30378.0, 51661.0], '10:1': [55227.0, 55228.0, 50771.0, 55227.0], '4:10': [34249.0, -2.0, 39397.0, 7321.0], '3:10': [39695.0, 39397.0, 35040.0, 38901.0], '3:9': [44646.0, 39595.0, 197.0, 38605.0], '2:9': [396.0, 0, -1.0, -1.0], '1:9': [31378.0, -1.0, -1.0, -1.0], '2:10': [296.0, 34842.0, 39395.0, 39496.0], '2:3': [37812.0, 49793.0, 55434.0, 53157.0], '2:4': [54146.0, 56427.0, 54148.0, 54049.0], '1:4': [47712.0, 51672.0, 47713.0, 55634.0], '1:3': [32168.0, 55633.0, 47712.0, 55433.0], '1:2': [27611.0, 28207.0, 29297.0, 33357.0], '1:1': [36323.0, 22367.0, 36323.0, 36621.0], '1:5': [55039.0, 50089.0, 53553.0, 54546.0], '1:6': [53553.0, 55735.0, 51972.0, 40192.0], '1:7': [54645.0, 52071.0, 55735.0, 56231.0], '1:8': [55735.0, 0, -1.0, 297.0], '2:6': [55437.0, 56627.0, 54645.0, 55437.0], '3:4': [55830.0, 55139.0, 56327.0, 55236.0], '2:1': [36522.0, 24249.0, 26621.0, 37909.0], '4:5': [55533.0, 46920.0, 49991.0, 13260.0], '3:5': [51869.0, 55536.0, 49596.0, 53651.0], '3:6': [54248.0, 51775.0, 55537.0, 55036.0], '3:7': [49992.0, 44646.0, 56528.0, 45933.0], '2:5': [55337.0, 56527.0, 54545.0, 54248.0], '2:7': [56329.0, 56727.0, 55636.0, 55735.0], '1:10': [97.0, 34841.0, 39395.0, 39396.0]}
            # 不是最优{'4:1': [29493.0, 31175.0, 31175.0, 30581.0], '3:1': [27316.0, 31275.0, 30483.0, 30186.0],
            #             '3:2': [31076.0, 30187.0, 31575.0, 31175.0], '4:2': [31174.0, 28800.0, 31176.0, 29691.0],
            #             '5:2': [30581.0, 29393.0, 30880.0, 30383.0], '5:3': [30384.0, 21674.0, -5.0, -5.0],
            #             '5:4': [21673.0, -4.0, 21675.0, -4.0], '5:5': [-5.0, -4.0, 21377.0, -3.0],
            #             '5:6': [-4.0, -3.0, -3.0, -3.0], '6:5': [-4.0, 291.0, -4.0, -5.0],
            #             '6:4': [-6.0, -5.0, 10190.0, 10089.0], '4:4': [16724.0, 25337.0, 28705.0, 18110.0],
            #             '4:3': [28801.0, 28407.0, 21673.0, 29492.0], '3:3': [31077.0, 29790.0, 18310.0, 28800.0],
            #             '6:3': [-6.0, 586.0, 29690.0, -6.0], '7:3': [-6.0, 9990.0, -6.0, -7.0],
            #             '7:2': [-6.0, 9890.0, -6.0, -7.0], '7:1': [-8.0, 9790.0, -7.0, -8.0],
            #             '6:2': [-8.0, 585.0, 30384.0, 1276.0], '8:3': [-7.0, -7.0, 8603.0, -6.0],
            #             '9:3': [-8.0, -8.0, -7.0, -8.0], '9:4': [-8.0, -7.0, -7.0, -7.0],
            #             '9:5': [-7.0, -7.0, -7.0, -8.0], '9:6': [-6.0, -6.0, -6.0, -6.0],
            #             '9:7': [-6.0, -5.0, -5.0, -6.0], '10:6': [-6.0, -6.0, -6.0, -7.0],
            #             '10:5': [-8.0, -7.0, -7.0, -7.0], '10:4': [-7.0, -7.0, -8.0, -7.0],
            #             '10:3': [-8.0, -7.0, -8.0, -7.0], '10:2': [-10.0, -8.0, 1175.0, -9.0],
            #             '10:7': [-6.0, -6.0, -6.0, -7.0], '8:7': [-5.0, -5.0, 687.0, -5.0],
            #             '7:7': [-4.0, -3.0, 1381.0, -4.0], '6:7': [-3.0, -4.0, 1778.0, -4.0],
            #             '6:8': [787.0, -3.0, -3.0, -4.0], '7:8': [-4.0, -5.0, 93.0, -5.0],
            #             '8:8': [-5.0, -5.0, -5.0, -5.0], '8:9': [-5.0, -6.0, -5.0, -7.0],
            #             '9:8': [-5.0, -6.0, -6.0, -6.0], '9:9': [-6.0, -7.0, -6.0, -6.0],
            #             '10:9': [-7.0, -8.0, -7.0, -7.0], '10:10': [-8.0, -8.0, 3852.0, -8.0],
            #             '9:10': [-6.0, -7.0, 3952.0, -8.0], '8:10': [-6.0, -6.0, 4052.0, -7.0],
            #             '10:8': [-7.0, -6.0, -6.0, -6.0], '7:10': [-5.0, -5.0, 4152.0, -5.0],
            #             '7:9': [-5.0, -5.0, -4.0, -5.0], '6:10': [-4.0, -4.0, 4252.0, -4.0],
            #             '6:9': [-4.0, -3.0, 392.0, -5.0], '5:9': [-3.0, -3.0, 888.0, -3.0],
            #             '5:10': [-3.0, -3.0, 4352.0, -4.0], '5:8': [-3.0, -3.0, -2.0, -3.0],
            #             '4:9': [-2.0, -2.0, 1285.0, -2.0], '4:8': [-1.0, -2.0, 6632.0, -2.0],
            #             '3:8': [6829.0, 0, 7128.0, -1.0], '4:7': [-2.0, 6532.0, 24451.0, -3.0],
            #             '5:7': [-4.0, -3.0, 1878.0, -2.0], '8:6': [-6.0, -5.0, 587.0, -5.0],
            #             '7:6': [-5.0, 1083.0, -4.0, -5.0], '6:6': [-4.0, 1183.0, -4.0, -4.0],
            #             '4:6': [21377.0, -2.0, -2.0, -3.0], '7:5': [-5.0, -5.0, -5.0, -5.0],
            #             '8:5': [-7.0, -6.0, -6.0, -7.0], '9:2': [-9.0, -8.0, 9591.0, -8.0],
            #             '8:2': [-7.0, 8404.0, 9691.0, -7.0], '8:4': [-7.0, -6.0, -6.0, -7.0],
            #             '7:4': [-6.0, -5.0, 10090.0, -6.0], '5:1': [27116.0, 30879.0, -7.0, 6818.0],
            #             '6:1': [-8.0, -7.0, 27116.0, 6819.0], '2:2': [29889.0, 31675.0, 29297.0, 30980.0],
            #             '8:1': [-8.0, -8.0, -8.0, -8.0], '9:1': [-9.0, 9491.0, -8.0, -8.0],
            #             '10:1': [-10.0, -9.0, 9391.0, -10.0], '4:10': [-2.0, -2.0, 4452.0, -2.0],
            #             '3:10': [6334.0, -1.0, -1.0, -1.0], '3:9': [6434.0, -1.0, -1.0, 0],
            #             '2:9': [99.0, 0, -1.0, -1.0], '1:9': [98.0, -1.0, -1.0, -1.0], '2:10': [-1.0, 0, 0, 6234.0],
            #             '2:3': [30189.0, 31775.0, 27217.0, 25930.0], '2:4': [30190.0, 31776.0, 31081.0, 29790.0],
            #             '1:4': [27217.0, 31577.0, 15241.0, 31080.0], '1:3': [29297.0, 27216.0, 25138.0, 21478.0],
            #             '1:2': [24443.0, 28207.0, 29297.0, 29694.0], '1:1': [17515.0, 22367.0, 17515.0, 27612.0],
            #             '1:5': [31081.0, 31479.0, 31577.0, 31875.0], '1:6': [31280.0, 31379.0, 24747.0, 31480.0],
            #             '1:7': [31479.0, 16233.0, 17818.0, 27323.0], '1:8': [31379.0, 0, -1.0, 198.0],
            #             '2:6': [29103.0, 32075.0, 31479.0, 28608.0], '3:4': [29791.0, 28706.0, 22469.0, 27813.0],
            #             '2:1': [30483.0, 24249.0, 26621.0, 30880.0], '4:5': [27813.0, 20980.0, 18608.0, 13260.0],
            #             '3:5': [28705.0, 20292.0, 29994.0, 27812.0], '3:6': [29201.0, 22372.0, 28609.0, 18606.0],
            #             '3:7': [19401.0, 7028.0, 31085.0, 20688.0], '2:5': [30389.0, 31975.0, 31478.0, 28409.0],
            #             '2:7': [31183.0, 32175.0, 30290.0, 27124.0], '1:10': [-2.0, -2.0, -2.0, 6134.0]}
        self.canvas = None
        self.root = None
        self.upsilon = 1
        self.quit = True
        self.startPos = [(10.5, 10.5), (1.5, 10.5), (10.5, 1.5), (1.5, 1.5)]
        self.start = 0
        self.end = 0
        self.timelist = []

    def updateQTable(self, reward, current_state):
        """Change q_table to reflect what we have learnt."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        # old_q = self.q_table[self.prev_s][self.prev_a]
        maxvalue = max(self.q_table[current_state])
        new_q = maxvalue + reward

        # # check if convergence
        # if new_q - self.q_table[self.prev_s][self.prev_a] > self.upsilon:
        #     self.quit = False
        # else:
        #     if self.startPos is not None:
        #         print(self.startPos)
        #         self.quit = False
        #         x, y = self.startPos[0]
        #         self.startPos.remove((x, y))
        #         my_mission.startAtWithPitchAndYaw(x, 227.0, y, 20.0, 90.0)

        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def updateQTableFromTerminatingState(self, reward):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        new_q = reward + self.q_table[self.prev_s][self.prev_a]
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.prev_a] = new_q

    def act(self, world_state, agent_host, current_r, world_x, world_y):
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

        self.drawQ(world_x, world_y, curr_x=int(obs[u'XPos']), curr_y=int(obs[u'ZPos']))

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

    def run(self, agent_host, world_x, world_y):
        """run the agent on the world"""
        self.start = time.time()
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
                        total_reward += self.act(world_state, agent_host, current_r, world_x, world_y)
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
                        total_reward += self.act(world_state, agent_host, current_r, world_x, world_y)
                        break
                    if not world_state.is_mission_running:
                        break

        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState(current_r)

        self.drawQ(world_x, world_y)
        self.end = time.time()
        return total_reward

    def drawQ(self, world_x, world_y, curr_x=None, curr_y=None):
        scale = 40
        # world_x = 10
        # world_y = 10
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x * scale, height=world_y * scale, borderwidth=0,
                                    highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [(0.5, action_inset), (0.5, 1 - action_inset), (action_inset, 0.5), (1 - action_inset, 0.5)]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x, y)
                self.canvas.create_rectangle(x * scale, y * scale, (x + 1) * scale, (y + 1) * scale, outline="#fff",
                                             fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = 255 * (value - min_value) / (max_value - min_value)  # map value to 0-255
                    color = int(color)
                    color = max(min(color, 255), 0)  # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255 - color, color, 0)
                    self.canvas.create_oval((x + action_positions[action][0] - action_radius) * scale,
                                            (y + action_positions[action][1] - action_radius) * scale,
                                            (x + action_positions[action][0] + action_radius) * scale,
                                            (y + action_positions[action][1] + action_radius) * scale,
                                            outline=color_string, fill=color_string)
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval((curr_x + 0.5 - curr_radius) * scale,
                                    (curr_y + 0.5 - curr_radius) * scale,
                                    (curr_x + 0.5 + curr_radius) * scale,
                                    (curr_y + 0.5 + curr_radius) * scale,
                                    outline="#fff", fill="#fff")
        self.root.update()


# draw the un-separated maze
def DrawMazeBase(my_mission, length, width, block_type):
    # to draw a maze base that each block considered as a cell
    length += 1
    width += 1
    my_mission.drawCuboid(0, 226, 0, 0, 230, length, block_type)
    my_mission.drawCuboid(0, 226, 0, width, 230, 0, block_type)
    my_mission.drawCuboid(width, 226, 0, width, 230, length, block_type)
    my_mission.drawCuboid(width, 226, length, 0, 230, length, block_type)

    # x = random.randint(1, width_m)
    # z = random.randint(1, length_m)
    my_mission.drawBlock(2, 226, 8, 'gold_block')


# put resource randomly in the maze
def release_resource(item_no, length, width):
    # blcok_list = ['diamond_block', 'stone', 'wool', 'glod_ore', 'diamond_ore', 'iron_ore',
    #               'dragon_egg', 'carpet', 'hopper', 'carrots', 'beacon', 'cocoa', 'cake', 'reeds', 'lever',
    #               'wheat', 'piston', 'bed']
    block_list = ['diamond_block', 'iron_block', 'redstone_block', 'quartz_block', 'hay_block']
    # item_list = random.sample(blcok_list, item_no)
    for i in range(item_no):
        length_random = random.randint(1, length - 1)
        width_random = random.randint(1, width - 1)
        # my_mission.drawItem(length_random, 227, width_random, item_list[i])
        for j in range(item_no):
            my_mission.drawBlock(length_random, 226, width_random, block_list[i])


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

DrawMazeBase(my_mission, length, width, "stonebrick")
# release_resource(resource_no, length, width)
reset = True
# if reset:
# my_mission.forceWorldReset()

# Attempt to start a mission
max_retries = 3
cumulative_rewards = []
total_repeat = 1000
# start learning
for i in range(total_repeat):
    print(len(agent.startPos))
    if i % 50 == 0:
        print(agent.q_table)
    if len(agent.timelist) == 10:
        differ = np.var(agent.timelist)
        print('The difference is:', differ)
        agent.timelist.clear()
        if differ < agent.upsilon:
            if len(agent.startPos) > 0:
                x, y = agent.startPos[0]
                agent.startPos.remove((x, y))
                my_mission.startAtWithPitchAndYaw(x, 227.0, y, 20.0, 90.0)
            else:
                print('Converged!!!')
                break

    agent.quit = True
    print()
    print('Repeat %d of %d' % (i + 1, total_repeat))
    my_mission_record = MalmoPython.MissionRecordSpec()
    # change the start position to get a more throughout table
    # if i > 1:
    #     my_mission.startAtWithPitchAndYaw(10.5, 227.0, 10.5, 20.0, 90.0)

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

    cumulative_reward = agent.run(agent_host, length, width)
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [cumulative_reward]
    agent.timelist.append(agent.end - agent.start)
    print('Running time for this epsoid is ', agent.end - agent.start)
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
