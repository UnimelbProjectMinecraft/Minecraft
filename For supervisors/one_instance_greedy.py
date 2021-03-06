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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER  DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------
def safeStartMission(agent_host, mission, client_pool, recording, role, experimentId):
    used_attempts = 0
    max_attempts = 5
    print("Calling startMission for role", role)
    while True:
        try:
            agent_host.startMission(mission, client_pool, recording, role, experimentId)
            break
        except MalmoPython.MissionException as e:
            errorCode = e.details.errorCode
            if errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_WARMING_UP:
                print("Server not quite ready yet - waiting...")
                time.sleep(2)
            elif errorCode == MalmoPython.MissionErrorCode.MISSION_INSUFFICIENT_CLIENTS_AVAILABLE:
                print("Not enough available Minecraft instances running.")
                used_attempts += 1
                if used_attempts < max_attempts:
                    print("Will wait in case they are starting up.", max_attempts - used_attempts, "attempts left.")
                    time.sleep(2)
            elif errorCode == MalmoPython.MissionErrorCode.MISSION_SERVER_NOT_FOUND:
                print("Server not found - has the mission with role 0 been started yet?")
                used_attempts += 1
                if used_attempts < max_attempts:
                    print("Will wait and retry.", max_attempts - used_attempts, "attempts left.")
                    time.sleep(2)
            else:
                print("Other error:", e.message)
                print("Waiting will not help here - bailing immediately.")
                exit(1)
        if used_attempts == max_attempts:
            print("All chances used up - bailing now.")
            exit(1)
    print("startMission called okay.")


def safeWaitForStart(agent_hosts):
    print("Waiting for the mission to start", end=' ')
    start_flags = [False for a in agent_hosts]
    start_time = time.time()
    time_out = 120  # Allow two minutes for mission to start.
    while not all(start_flags) and time.time() - start_time < time_out:
        states = [a.peekWorldState() for a in agent_hosts]
        start_flags = [w.has_mission_begun for w in states]
        errors = [e for w in states for e in w.errors]
        if len(errors) > 0:
            print("Errors waiting for mission start:")
            for e in errors:
                print(e.text)
            print("Bailing now.")
            exit(1)
        time.sleep(0.1)
        print(".", end=' ')
    print()
    if time.time() - start_time >= time_out:
        print("Timed out waiting for mission to begin. Bailing.")
        exit(1)
    print("Mission has started.")


class Qlearning(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self):
        self.epsilon = 0.01  # chance of taking a random action instead of the best
        self.goalNo = 4
        # let say there's four different goals(Goal 0, subGoal 1, subGoal 2, subGoal 3)
        self.goalSelection = '0100'
        self.goalConverge = [False, False, False, False]
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
        self.q_table = {'4:1': [[2625663.0, 102986.0, 48005.0, 2628982.0], [10996.0, 816.0, 57777.0, 814.0],
                                [416815.0, 575521.0, 402960.0, 425488.0], [320443.0, 320444.0, 1173223.0, 1218452.0]],
                        '3:1': [[-7.0, 2388390.0, -7.0, -8.0], [34387.0, 58687.0, -2.0, 10996.0],
                                [562801.0, 428803.0, 425485.0, 569111.0], [293524.0, 320441.0, 375477.0, 1188133.0]],
                        '5:1': [[2621637.0, 2630792.0, 2618182.0, 1279224.0], [156957.0, 827069.0, -2.0, -3.0],
                                [-3.0, 569113.0, -4.0, -4.0], [509224.0, 1233862.0, 1173224.0, 514159.0]],
                        '5:2': [[2619110.0, 5638692.0, 2592065.0, 184087.0], [291492.0, 301703.0, 956640.0, 1043387.0],
                                [430615.0, 588741.0, -3.0, 506.0], [1233861.0, 1264981.0, 517478.0, 977399.0]],
                        '6:1': [[1279224.0, 2621637.0, -7.0, -8.0], [-2.0, 766034.0, -3.0, -3.0],
                                [-3.0, -3.0, 430615.0, -3.0], [-7.0, 977399.0, 254199.0, -6.0]],
                        '6:2': [[59524.0, 4273160.0, 2637119.0, -6.0], [-3.0, 1420994.0, 956641.0, 458024.0],
                                [-3.0, 414972.0, -3.0, -2.0], [-7.0, 977400.0, -6.0, 977396.0]],
                        '4:2': [[102987.0, 2619111.0, -7.0, 59526.0], [31477.0, 1494622.0, 4153.0, 956641.0],
                                [-3.0, 415699.0, 415288.0, 582031.0], [155626.0, 11130.0, 155626.0, 519288.0]],
                        '7:1': [[-7.0, -7.0, 2619109.0, 8129787.0], [-2.0, 1151740.0, -2.0, -2.0],
                                [-3.0, -3.0, -3.0, -4.0], [-6.0, 37625.0, -5.0, -6.0]],
                        '7:2': [[7807005.0, 8202804.0, 2637118.0, 11574235.0], [-2.0, 1393721.0, -2.0, -2.0],
                                [-3.0, 56160.0, -3.0, -2.0], [-5.0, -5.0, -5.0, 1044442.0]],
                        '7:3': [[8202805.0, 3089738.0, 2637119.0, 1353860.0],
                                [596523.0, 1226676.0, 1430312.0, 623441.0], [38141.0, 68888.0, -2.0, 56159.0],
                                [506476.0, -4.0, 1173226.0, 977398.0]],
                        '7:4': [[3089737.0, 1731832.0, 18555.0, 11762873.0],
                                [1425602.0, 1264993.0, 1061096.0, 1226675.0], [-1.0, 80507.0, -1.0, -2.0],
                                [-3.0, 144556.0, 155629.0, -4.0]],
                        '7:5': [[8690726.0, 12583331.0, -128846.0, 4901530.0],
                                [1390402.0, 1264992.0, -130201.0, 1188956.0], [-2.0, -1.0, 87117.0, -1.0],
                                [-4.0, 1607.0, -11973.0, 1114590.0]],
                        '2:1': [[-7.0, -6.0, 2438209.0, -7.0], [-2.0, 1477192.0, -1.0, -1.0],
                                [562800.0, 428802.0, -5.0, 562801.0], [1173222.0, 320440.0, 375476.0, 1188132.0]],
                        '3:2': [[-6.0, -6873.0, 2388391.0, 48007.0], [12706.0, 58688.0, 12706.0, 1488712.0],
                                [556592.0, -191.0, -5.0, 415289.0], [446723.0, -11882.0, 254195.0, 155627.0]],
                        '1:1': [[13443781.0, 13651218.0, 2363782.0, 2388390.0], [-1.0, -2.0, -2.0, 1471582.0],
                                [550481.0, 402957.0, 427093.0, 562800.0], [1173221.0, 320439.0, 1058149.0, 1173222.0]],
                        '2:2': [[-7.0, 664782.0, 13599209.0, 1000.0], [-2.0, -1.0, 0, 1482902.0],
                                [428803.0, -4.0, 427092.0, -4.0], [-8.0, -7.0, 293521.0, 320441.0]],
                        '1:2': [[1669703.0, 13703328.0, 13547300.0, 13340663.0], [-2.0, -1.0, -2.0, -1.0],
                                [-6.0, -5.0, -5.0, 428802.0], [446721.0, -7.0, -7.0, 216068.0]],
                        '2:3': [[-6.0, 3210836.0, -5.0, -5182.0], [0, 0, -1.0, 2109.0], [333004.0, -4.0, -4.0, -391.0],
                                [-7.0, -6.0, -7.0, -2882.0]],
                        '1:3': [[13340664.0, 13755538.0, 13340665.0, 664782.0], [-1.0, -1.0, -1.0, -1.0],
                                [-4.0, -4.0, -5.0, -4.0], [320439.0, -6.0, -6.0, -6.0]],
                        '1:4': [[3222445.0, 13807948.0, 3210837.0, 3210836.0], [-1.0, -1.0, -1.0, -1.0],
                                [-4.0, -3.0, -3.0, -3.0], [-6.0, -6.0, -6.0, -6.0]],
                        '1:5': [[13703329.0, 13807949.0, 3210838.0, 610746.0], [-1.0, -2.0, -1.0, -1.0],
                                [-3.0, -3.0, -3.0, -3.0], [-5.0, -5.0, -5.0, -6.0]],
                        '1:6': [[3176611.0, 13807950.0, 13807949.0, -2.0], [-2.0, -4.0, -3.0, -2.0],
                                [-3.0, -3.0, -4.0, 44402.0], [-6.0, 9820.0, -6.0, -5.0]],
                        '2:6': [[-3.0, 13340670.0, -3.0, -2.0], [-2.0, -3.0, -2.0, 184949.0],
                                [-3.0, -3.0, -3.0, 56721.0], [509225.0, 509225.0, 5710.0, 557916.0]],
                        '3:6': [[-3.0, 13913070.0, -2.0, 2103283.0], [188659.0, -3.0, -2.0, -2.0],
                                [-3.0, -2.0, -2.0, 96076.0], [509226.0, -3.0, 509226.0, 665606.0]],
                        '4:6': [[3928291.0, 12000927.0, 1242514.0, 9726866.0], [153359.0, -2.0, -2.0, -2.0],
                                [-1.0, 449688.0, -2.0, 550488.0], [461437.0, 200332.0, 509227.0, 1203248.0]],
                        '4:5': [[2619110.0, 3914.0, 130426.0, 4273163.0], [156069.0, -2.0, 6916.0, -2.0],
                                [-2.0, 470543.0, -1.0, 478463.0], [91766.0, 1058157.0, 9822.0, -3.0]],
                        '5:5': [[2619113.0, 9726866.0, 2892453.0, -68737.0], [-1.0, -2.0, 17733.0, -12573.0],
                                [562806.0, 544479.0, 478462.0, 609471.0], [1173228.0, 1280693.0, 510136.0, -118065.0]],
                        '5:4': [[5638692.0, 6637900.0, 2219230.0, 137633.0], [1096916.0, -2.0, 0, 1008269.0],
                                [-1.0, 602461.0, 463025.0, -1.0], [538361.0, 1264983.0, 1264981.0, 554206.0]],
                        '6:4': [[2618184.0, -8964.0, 18556.0, -4.0], [1226678.0, -105710.0, 973750.0, 781042.0],
                                [0, 11236.0, -1.0, -1.0], [1173226.0, -17773.0, 557916.0, 144555.0]],
                        '6:3': [[2637118.0, -5.0, 5638692.0, -5.0], [1226677.0, 1226677.0, 1435122.0, 1133332.0],
                                [-1.0, 10926.0, 414973.0, 65369.0], [514160.0, 557915.0, 1233863.0, 977399.0]],
                        '8:3': [[10316083.0, 8093481.0, 8202804.0, 12682844.0], [-1.0, 1416482.0, 623442.0, -1.0],
                                [-3.0, -3.0, 56160.0, -3.0], [515767.0, 951379.0, 1058153.0, 506477.0]],
                        '9:3': [[12682843.0, 12682845.0, 12241562.0, 12387389.0], [-3.0, -3.0, 1412072.0, -3.0],
                                [-4.0, 29330.0, 29332.0, -5.0], [-4.0, 155624.0, 1030834.0, -4.0]],
                        '5:3': [[4273160.0, 6421036.0, 2619720.0, 5638691.0],
                                [956641.0, 1008268.0, 1440032.0, 827071.0], [422571.0, 595551.0, -2.0, 414972.0],
                                [1233862.0, 1264982.0, 1173226.0, 1218453.0]],
                        '4:4': [[2619111.0, -4.0, 3720.0, 2637121.0], [0, 44460.0, 192469.0, 0],
                                [-2.0, 419246.0, -1.0, 575524.0], [-6.0, 1058156.0, 547387.0, 1264982.0]],
                        '3:4': [[-391.0, -3.0, 2363784.0, 3719.0], [200388.0, 148242.0, 103697.0, 32252.0],
                                [-91.0, 69440.0, -4.0, -2.0], [-19273.0, -4.0, -5.0, 1264981.0]],
                        '8:2': [[10975117.0, 12241562.0, 8202805.0, 12338680.0], [-2.0, -2.0, -2.0, -3.0],
                                [-4.0, -3.0, 3705.0, -4.0], [-4.0, 1058152.0, 507194.0, -5.0]],
                        '9:2': [[12436197.0, 12682844.0, 12338679.0, 11480517.0], [-3.0, 1407762.0, -3.0, -3.0],
                                [-4.0, -4.0, -4.0, 443863.0], [-5.0, 557911.0, -5.0, -6.0]],
                        '10:2': [[12485105.0, 12583325.0, 9892794.0, 11480517.0], [-3.0, -4.0, 1403552.0, -3.0],
                                 [-6.0, 501499.0, -5.0, 443863.0], [990603.0, 990605.0, 557910.0, 951377.0]],
                        '10:1': [[12338678.0, 12485106.0, 12338679.0, 11857788.0], [-4.0, 1399442.0, -4.0, -4.0],
                                 [154742.0, 478453.0, 154741.0, 438944.0], [665597.0, 990604.0, -7.0, 665597.0]],
                        '10:3': [[12583324.0, 12338681.0, 12682844.0, 12583325.0], [-3.0, -3.0, -3.0, -3.0],
                                 [213297.0, 501500.0, 12013.0, 443864.0], [990604.0, 990606.0, 977397.0, 665599.0]],
                        '10:4': [[4370201.0, 9768169.0, 12682845.0, 9644641.0], [-3.0, -3.0, 4129.0, -3.0],
                                 [438946.0, 506510.0, -3.0, 154745.0], [938969.0, 1072061.0, 506476.0, 591883.0]],
                        '10:5': [[10840793.0, 9768170.0, 12096838.0, 7631465.0], [-3.0, 291486.0, -3.0, -3.0],
                                 [478455.0, 511620.0, -2.0, 478456.0], [659191.0, 964290.0, 1072062.0, 591884.0]],
                        '10:6': [[5092000.0, 9768171.0, 11621249.0, 8387955.0], [291485.0, -4.0, 781036.0, 291486.0],
                                 [-2.0, -2.0, 516830.0, -2.0], [659192.0, 1003918.0, -3.0, 154863.0]],
                        '9:6': [[12241565.0, 12338683.0, 12832875.0, 11574239.0],
                                [410987.0, 1226671.0, 751221.0, 751219.0], [-2.0, -2.0, 522140.0, -2.0],
                                [-3.0, 1003919.0, 154867.0, 108418.0]],
                        '9:7': [[12338684.0, 9320172.0, 12338684.0, 10840796.0],
                                [1170243.0, 1170243.0, 1384962.0, 1078898.0], [6414.0, 92674.0, -2.0, -3.0],
                                [591886.0, 659194.0, 1114592.0, 659194.0]],
                        '8:7': [[12338685.0, 10619754.0, 10840799.0, 11715566.0],
                                [990956.0, 1078897.0, 1386172.0, 1226671.0], [-2.0, -2.0, 98486.0, -1.0],
                                [707558.0, 799757.0, 1143711.0, 640568.0]],
                        '9:5': [[12682845.0, 12782765.0, 12338684.0, 12096837.0], [-3.0, 271469.0, 906412.0, 252249.0],
                                [-2.0, -2.0, -2.0, -2.0], [15371.0, 964291.0, 1086072.0, 1072061.0]],
                        '9:4': [[12338681.0, 12782764.0, 11294883.0, 12338681.0], [-2.0, -3.0, 1395528.0, -2.0],
                                [29331.0, -2.0, -3.0, -3.0], [506477.0, 3431.0, 144554.0, -4.0]],
                        '8:4': [[11248773.0, 8690728.0, 8021162.0, 11810283.0], [28485.0, 494451.0, 1420992.0, -3.0],
                                [-3.0, 20625.0, -2.0, -2.0], [1044443.0, -3.0, 144555.0, -5.0]],
                        '8:5': [[10840793.0, 12534121.0, 9320171.0, 10975122.0],
                                [354942.0, 365951.0, 1264993.0, 470032.0], [-2.0, -2.0, 74598.0, -3.0],
                                [507195.0, 1114591.0, 63366.0, 175783.0]],
                        '8:6': [[12000921.0, 12338684.0, 12883085.0, 12732756.0],
                                [906412.0, 583412.0, 990957.0, 751220.0], [-2.0, -2.0, 527550.0, 193384.0],
                                [1114590.0, 1129101.0, 1607.0, -1.0]],
                        '8:8': [[10445417.0, 10402208.0, 11527335.0, 9481608.0],
                                [973746.0, 344028.0, 874290.0, 1170243.0], [-2.0, 213300.0, 434532.0, -3.0],
                                [1003920.0, -1.0, 612324.0, 134448.0]],
                        '7:8': [[12883086.0, 10751981.0, 12883088.0, 10751981.0], [637049.0, -5.0, -4.0, 973745.0],
                                [436642.0, -2.0, -3.0, 434531.0], [990612.0, 809066.0, 938974.0, 764221.0]],
                        '6:8': [[-1580797.0, 11248781.0, 12883089.0, 12883087.0], [-3991.0, -3.0, -4.0, -4.0],
                                [-3191.0, -2.0, -2.0, 415697.0], [59354.0, 828383.0, 926864.0, 951384.0]],
                        '6:9': [[12387398.0, 9603736.0, 8962796.0, 10102846.0], [-3.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -3.0, -2.0], [730785.0, -1.0, -2.0, 938974.0]],
                        '5:9': [[8962797.0, 4971261.0, 12933498.0, 11294890.0], [-4.0, -3.0, -5.0, -3.0],
                                [-2.0, -3.0, 478458.0, -3.0], [938973.0, 915053.0, 870017.0, 859310.0]],
                        '5:8': [[12633043.0, 12832879.0, 12883090.0, 12883088.0], [-3.0, -4.0, -3.0, -4.0],
                                [459516.0, -2.0, 466732.0, -2.0], [258596.0, 195504.0, 848698.0, 951383.0]],
                        '5:10': [[6859965.0, -2.0, 4971262.0, 6421035.0], [-4.0, -4.0, -5.0, -5.0],
                                 [-3.0, -3.0, -4.0, -3.0], [915054.0, 848699.0, -3.0, 747299.0]],
                        '4:3': [[-7.0, -5.0, -7964.0, 5638692.0], [1382851.0, 103697.0, 1500632.0, 1264997.0],
                                [-2.0, -2.0, -291.0, 416209.0], [-5.0, 155629.0, -17273.0, 1233863.0]],
                        '5:6': [[9726865.0, 11621254.0, 3928290.0, 413907.0], [-2.0, -1.0, -2.0, -1.0],
                                [588743.0, 522142.0, 0, 538669.0], [1249374.0, 870022.0, 755714.0, 1296603.0]],
                        '5:7': [[11248781.0, 10840802.0, 12984007.0, -114401.0], [-2.0, -3.0, -2.0, -3591.0],
                                [527552.0, -3.0, 491787.0, -14664.0], [915059.0, 182296.0, 182296.0, 258597.0]],
                        '4:7': [[9726865.0, 13085326.0, 9240061.0, 11621254.0], [150749.0, -3.0, -3.0, -3.0],
                                [459516.0, 449687.0, 102983.0, 522142.0], [-3.0, 828380.0, 21142.0, 204242.0]],
                        '3:7': [[13913069.0, 13965880.0, 5833760.0, 9240060.0], [-2.0, -3.0, -2.0, 145830.0],
                                [-3.0, -3.0, -3.0, 452797.0], [-4.0, -3.0, -4.0, 204241.0]],
                        '3:8': [[13495498.0, 11905410.0, 14018890.0, 12682854.0], [-3.0, -4.0, -18064.0, -4.0],
                                [-3.0, 456103.0, -4391.0, 478459.0], [200331.0, -2.0, -491.0, 915052.0]],
                        '4:8': [[12000927.0, 12682853.0, 13136136.0, 12583335.0], [143521.0, -4.0, -4.0, -4.0],
                                [487178.0, 478458.0, 474449.0, 459515.0], [790543.0, 915053.0, -2.0, 938973.0]],
                        '4:9': [[13034617.0, 9726867.0, 11574247.0, 12832879.0], [122302.0, -5.0, -4.0, -4.0],
                                [482768.0, -4.0, -4.0, 474448.0], [870016.0, 790544.0, 828380.0, 938972.0]],
                        '3:9': [[13289363.0, 3234161.0, 13238152.0, 11065847.0], [-4.0, -5.0, -4.0, 6911.0],
                                [456104.0, -5.0, 463020.0, 482767.0], [828379.0, 755707.0, 838488.0, 926862.0]],
                        '4:10': [[11202774.0, 6892174.0, -2.0, 4470054.0], [-4.0, -5.0, -5.0, -5.0],
                                 [63028.0, -4.0, -4.0, -4.0], [915053.0, -4.0, -5.0, 37659.0]],
                        '3:5': [[-3.0, 6724.0, -2.0, 2103284.0], [196378.0, -1.0, 0, 0], [-2.0, -3.0, -2.0, 203903.0],
                                [-4.0, 557916.0, -6.0, -4.0]],
                        '9:1': [[12000916.0, 12682843.0, 11248771.0, 11294880.0], [-4.0, -4.0, -3.0, -3.0],
                                [-5.0, -5.0, -5.0, 154742.0], [-6.0, -6.0, -6.0, 591880.0]],
                        '8:1': [[10975117.0, 10316083.0, 3132371.0, 11621244.0], [-2.0, -3.0, -3.0, -2.0],
                                [-4.0, -4.0, -4.0, -4.0], [-6.0, -5.0, -6.0, -6.0]],
                        '2:4': [[9138.0, 16356.0, 13495493.0, 3720.0], [0, 0, 0, 196378.0], [-3.0, -3.0, -3.0, 2123.0],
                                [-7.0, -6.0, -6.0, 91765.0]],
                        '2:7': [[12984006.0, 268862.0, 2116.0, 13913070.0], [181339.0, -14773.0, -4.0, 57766.0],
                                [-3.0, -991.0, -3.0, 82556.0], [509226.0, -12773.0, -5.0, -3.0]],
                        '2:5': [[664783.0, -2.0, -3.0, 6723.0], [-1.0, -2.0, -1.0, -1.0], [-3.0, -3.0, -3.0, -3.0],
                                [-6.0, -5.0, -6.0, 509226.0]],
                        '10:7': [[9768170.0, 5067492.0, 11668357.0, 244349.0], [-4.0, 97375.0, 1151735.0, -4.0],
                                 [183574.0, -4.0, -3.0, 21948.0], [-2.0, -3.0, 1086073.0, -2.0]],
                        '9:8': [[10840797.0, 4833404.0, 8845467.0, 4531276.0],
                                [1383852.0, 1151733.0, 1151733.0, 990952.0], [6413.0, 92673.0, 434531.0, -4.0],
                                [964292.0, 659193.0, 207912.0, 601801.0]],
                        '10:8': [[-4.0, 6240280.0, 3038697.0, 7926.0], [874290.0, 233729.0, 1151734.0, -5.0],
                                 [173864.0, -4.0, -4.0, -5.0], [-2.0, 640565.0, 591886.0, 601801.0]],
                        '9:9': [[9851087.0, 2980741.0, 6240282.0, 3535064.0],
                                [1362833.0, 990949.0, 1207759.0, 1207759.0], [415695.0, -5.0, 415695.0, 415693.0],
                                [659194.0, 56256.0, 607013.0, 652883.0]],
                        '8:9': [[10445418.0, 5216548.0, 8690729.0, 5216548.0], [664563.0, 678670.0, 90468.0, 1207760.0],
                                [434531.0, -4.0, -3.0, -3.0], [634559.0, -2.0, 596696.0, -1.0]],
                        '7:9': [[12000925.0, 8614208.0, 10102845.0, 9200147.0], [609930.0, -5.0, -4.0, -5.0],
                                [-3.0, -3.0, -2.0, -4.0], [951384.0, 678921.0, 9559.0, 634558.0]],
                        '6:10': [[9603737.0, 3478731.0, 6421034.0, 4123090.0], [-4.0, -5.0, -5.0, -6.0],
                                 [-3.0, -3.0, -4.0, -3.0], [108421.0, -2.0, 764218.0, 596695.0]],
                        '7:10': [[10102846.0, 290594.0, 8614209.0, 290595.0], [583411.0, -6.0, -5.0, -6.0],
                                 [-3.0, -4.0, -3.0, -4.0], [678922.0, 596695.0, 596694.0, -2.0]],
                        '10:9': [[3028787.0, 3058812.0, 6892168.0, 2999659.0],
                                 [939625.0, 1133322.0, 1342923.0, 1207759.0], [21947.0, 213297.0, 415694.0, 118508.0],
                                 [92902.0, 634555.0, 659193.0, 634556.0]],
                        '10:10': [[3058813.0, 3058812.0, 2935196.0, 2962222.0],
                                  [1323213.0, 1025768.0, 890295.0, 1133322.0], [415693.0, 213297.0, -6.0, -6.0],
                                  [646674.0, 634555.0, 523304.0, 634555.0]],
                        '9:10': [[2776408.0, 2776407.0, 3638028.0, 2876332.0],
                                 [494446.0, 494445.0, 858378.0, 1025768.0], [-5.0, -5.0, -5.0, -6.0],
                                 [569840.0, 56256.0, -2.0, -2.0]],
                        '8:10': [[8614212.0, 2999660.0, 290594.0, 2776407.0], [874288.0, -6.0, -6.0, -6.0],
                                 [-4.0, -5.0, -4.0, -4.0], [12168.0, -2.0, -2.0, -2.0]],
                        '7:6': [[11480523.0, 12883086.0, 12096840.0, 12782766.0],
                                [1388892.0, 1245782.0, 990956.0, 973747.0], [66680.0, -1.0, 533060.0, 0],
                                [640566.0, 1143711.0, 1264985.0, 1100282.0]],
                        '6:6': [[-3382.0, -49064.0, 10840800.0, 12338686.0], [-4591.0, -24873.0, -2.0, 1245783.0],
                                [213306.0, -2891.0, 550488.0, 441361.0], [-129065.0, 1312613.0, 1264984.0, 1264984.0]],
                        '1:7': [[3210839.0, 3222450.0, 13807950.0, 13860460.0], [-3.0, -4.0, -4.0, 177829.0],
                                [-4.0, -4.0, -4.0, -4.0], [9819.0, -6.0, 5711.0, 509225.0]],
                        '1:8': [[3222449.0, 4470053.0, 76425.0, 154753.0], [177828.0, -6.0, -5.0, -11282.0],
                                [-4.0, -6.0, -5.0, -4091.0], [509224.0, -6.0, -6.0, -5173.0]],
                        '7:7': [[12682849.0, 12883087.0, -907848.0, 12338684.0],
                                [1387482.0, 874290.0, -98837.0, 1226672.0], [441361.0, -1.0, -9073.0, -2.0],
                                [1158421.0, 951384.0, 374377.0, 838494.0]],
                        '2:9': [[144154.0, 2489035.0, 2463526.0, 13238153.0], [-4191.0, -6.0, -6.0, -5.0],
                                [-14464.0, 169660.0, 169660.0, 478457.0], [-70628.0, 828378.0, 838487.0, 903543.0]],
                        '2:10': [[13238152.0, 425318.0, 3143287.0, 3397378.0], [-6.0, -6.0, -5.0, -6.0],
                                 [478456.0, 456101.0, -7.0, 463020.0], [880824.0, -6.0, 880822.0, 915051.0]],
                        '3:10': [[3199337.0, 496473.0, 12984006.0, 664789.0], [-5.0, -5.0, -6.0, -5.0],
                                 [474448.0, -5.0, -5.0, -5.0], [915052.0, 127147.0, 870014.0, 790544.0]],
                        '1:10': [[3176615.0, 3222452.0, 3176616.0, 12984006.0], [171108.0, -6.0, -6.0, -6.0],
                                 [146533.0, 139023.0, 466727.0, 466728.0], [848696.0, 903540.0, 870013.0, 903541.0]],
                        '1:9': [[2566161.0, 11433817.0, 2514644.0, 2566163.0], [174418.0, -6.0, -6.0, -6.0],
                                [-5.0, -6.0, -6.0, 466729.0], [9819.0, 838486.0, 828378.0, 870015.0]]}
        self.canvas = None
        self.root = None
        self.upsilon = 1
        self.quit = True
        self.startPos = [(10.5, 10.5), (1.5, 10.5), (10.5, 1.5), (1.5, 1.5)]
        self.start = 0
        self.end = 0
        self.timelist = []
        self.goalIndex = 1
        self.goalSelect()

    def goalSelect(self):
        switch = {'1000': 0, '0100': 1, '0010': 2, '0001': 3}
        self.goalIndex = switch.get(self.goalSelection)

    def getGoalNo(self):
        return self.goalIndex

    def updateQTable(self, reward, current_state):
        """Change q_table to reflect what we have learnt."""

        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        # old_q = self.q_table[self.prev_s][self.prev_a]
        maxvalue = max(self.q_table[current_state][self.goalIndex])
        new_q = maxvalue + reward
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.goalIndex][self.prev_a] = new_q
        # print('self.q_table[', self.prev_s, '][', self.goalIndex, '][', self.prev_a, '] = ', new_q)
        # print(self.q_table)

    def updateQTableFromTerminatingState(self, reward):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        new_q = reward + self.q_table[self.prev_s][self.goalIndex][self.prev_a]
        # assign the new action value to the Q-table
        self.q_table[self.prev_s][self.goalIndex][self.prev_a] = new_q

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
            self.q_table[current_s] = [([0] * len(self.actions)) for p in range(self.goalNo)]

        self.drawQ(world_x, world_y, curr_x=int(obs[u'XPos']), curr_y=int(obs[u'ZPos']))

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            m = max(self.q_table[current_s][self.goalIndex])
            self.logger.debug("Current values: %s" % ",".join(str(x) for x in self.q_table[current_s]))
            l = list()
            for x in range(0, len(self.actions)):
                if self.q_table[current_s][self.goalIndex][x] == m:
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

    def run(self, agent_host,agent_host1, world_x, world_y):
        """run the agent on the world"""
        self.start = time.time()
        total_reward = 0

        self.prev_s = None
        self.prev_s1 = None

        self.prev_a = None
        self.prev_a1 = None

        self.goalIndex = self.getGoalNo()
        is_first_action = True

        # main loop:
        world_state = agent_host.getWorldState()
        world_state1 = agent_host.getWorldState()
        while world_state.is_mission_running or world_state1.is_mission_running :

            current_r = 0
            current_r1= 0
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
                while True:
                    time.sleep(0.1)
                    world_state1 = agent_host1.getWorldState()
                    for error in world_state1.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state1.rewards:
                        current_r1 += reward.getValue()

                    if world_state1.is_mission_running and len(world_state1.observations) > 0 and not \
                            world_state1.observations[-1].text == "{}":
                        total_reward += self.act(world_state1, agent_host1, current_r1, world_x, world_y)
                        break
                    if not world_state1.is_mission_running:
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
                while world_state1.is_mission_running and current_r1 == 0:
                    time.sleep(0.1)
                    world_state1 = agent_host1.getWorldState()
                    for error in world_state1.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state1.rewards:
                        current_r1 += reward.getValue()                       
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
                while True:
                    time.sleep(0.1)
                    world_state1 = agent_host1.getWorldState()
                    for error in world_state1.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state1.rewards:
                        current_r1 += reward.getValue()
                    if world_state1.is_mission_running and len(world_state1.observations) > 0 and not \
                            world_state1.observations[-1].text == "{}":
                        total_reward += self.act(world_state1, agent_host1, current_r1, world_x, world_y)
                        break
                    if not world_state1.is_mission_running:
                        break
        # process final reward
            self.logger.debug("Final reward: %d" % current_r)
            total_reward += (current_r+current_r1)

        self.drawQ(world_x, world_y)
        self.end = time.time()
        return total_reward

    def drawQ(self, world_x, world_y, curr_x=None, curr_y=None):
        scale = 40

        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            # draw the background(base)
            self.canvas = tk.Canvas(self.root, width=(world_x + 2) * scale, height=(world_y + 2) * scale, borderwidth=0,
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
        # draw the chart
        for x in range(1, world_x + 1):
            for y in range(1, world_y + 1):
                s = "%d:%d" % (x, y)
                self.canvas.create_rectangle(x * scale, y * scale, (x + 1) * scale, (y + 1) * scale, outline="#ffe",
                                             fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][self.goalIndex][action]
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
agent_host1 = MalmoPython.AgentHost()
agent_host2 = MalmoPython.AgentHost()
# sys.argv receive the command arguments
try:
    agent_host1.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host1.getUsage())
    exit(1)

if agent_host1.receivedArgument("help"):
    print(agent_host1.getUsage())
    exit(0)

# get the size of maze from command line


# read xml from outside file
mission_file = './random_map_2.xml'
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
client_pool = MalmoPython.ClientPool()
client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10000))
client_pool.add(MalmoPython.ClientInfo('127.0.0.1', 10001))
safeStartMission(agent_host1, my_mission, client_pool, MalmoPython.MissionRecordSpec(), 0, '')
safeStartMission(agent_host2, my_mission, client_pool, MalmoPython.MissionRecordSpec(), 1, '')
safeWaitForStart([agent_host1, agent_host2])

# DrawMazeBase(my_mission, length, width, "stonebrick")
# release_resource(resource_no, length, width)
# reset = True
# if reset:
# my_mission.forceWorldReset()

# Attempt to start a mission
max_retries = 3
cumulative_rewards = []
total_repeat = 5

# start learning
for i in range(total_repeat):
    # print(len(agent.startPos))
    # if i % 50 == 0:
    #     print(agent.q_table)
    # if len(agent.timelist) == 10:
    #     differ = np.var(agent.timelist)
    #     print('The difference is:', differ)
    #     agent.timelist.clear()
    #     if differ < agent.upsilon:
    #         if len(agent.startPos) > 0:
    #             x, y = agent.startPos[0]
    #             agent.startPos.remove((x, y))
    #             my_mission.startAtWithPitchAndYaw(x, 227.0, y, 20.0, 90.0)
    #         else:
    #             print('Converged!!!')
    #             break

    # agent.quit = True
    # print()
    # print('Repeat %d of %d' % (i + 1, total_repeat))
    # my_mission_record = MalmoPython.MissionRecordSpec()
    # # change the start position to get a more throughout table
    # # if i > 1:
    # #     my_mission.startAtWithPitchAndYaw(10.5, 227.0, 10.5, 20.0, 90.0)

    # for retry in range(max_retries):
    #     try:
    #         agent_host1.startMission(my_mission, my_mission_record)
    #         agent_host2.startMission(my_mission, my_mission_record)
    #         break
    #     except RuntimeError as e:
    #         if retry == max_retries - 1:
    #             print("Error starting mission:", e)
    #             exit(1)
    #         else:
    #             time.sleep(2)

    # # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state1 = agent_host1.getWorldState()
    world_state2 = agent_host2.getWorldState()
    while not world_state1.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state1 = agent_host1.getWorldState()
        for error in world_state1.errors:
            print("Error:", error.text)
    while not world_state2.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state2 = agent_host2.getWorldState()
        for error in world_state2.errors:
            print("Error:", error.text)


    print()

    cumulative_reward = agent.run(agent_host1,agent_host2, 10, 10)
    # cumulative_reward = cumulative_reward1 + cumulative_reward2
    print('Cumulative reward: %d' % cumulative_reward)
    cumulative_rewards += [cumulative_reward]
    agent.timelist.append(agent.end - agent.start)
    print('Running time for this epsoid is ', agent.end - agent.start)
    time.sleep(0.5)

print('Q-table: ', agent.q_table)
print("Mission running ", end=' ')

# agent_host.sendCommand("move 1")
while agent_host1.peekWorldState().is_mission_running or agent_host2.peekWorldState().is_mission_running:
    # replace
    # while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state1 = agent_host1.getWorldState()
    world_state2 = agent_host2.getWorldState()

    for error in world_state1.errors:
        print("Error:", error.text)
    for error in world_state12.errors:
        print("Error:", error.text)

print()
print("Mission ended")
# Mission has ended.
