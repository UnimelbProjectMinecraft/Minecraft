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
        self.epsilon = 0.5  # chance of taking a random action instead of the best
        self.goalNo = 4
        # let say there's four different goals(Goal 0, subGoal 1, subGoal 2, subGoal 3)
        self.goalSelection = '1000'
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
        self.q_table = {'4:1': [[2625663.0, 102986.0, 48005.0, 2628982.0], [4744739.0, 4872966.0, 4703693.0, 4788283.0], [1933152.0, 1971471.0, 1990978.0, 3283573.0], [1624979.0, 1709642.0, 1330621.0, 1709642.0]], '3:1': [[2388389.0, 2388390.0, -7.0, -8.0], [4703693.0, 4788284.0, 4680265.0, 4680267.0], [3283571.0, 3292471.0, 1294299.0, 1990979.0], [1314340.0, 1312921.0, 1312921.0, 1624979.0]], '5:1': [[2621637.0, 2630792.0, 2618182.0, 1279224.0], [4788283.0, 4779373.0, 4843838.0, 4547085.0], [2693334.0, 3305536.0, 1857816.0, 1857816.0], [1709642.0, 1709643.0, 1709641.0, 1709641.0]], '5:2': [[2619110.0, 5638692.0, 2592065.0, 184087.0], [4779374.0, 4703693.0, 4824921.0, 4779372.0], [3283573.0, 3491179.0, 3254855.0, 1933153.0], [1709642.0, 1770116.0, 1709642.0, 1745587.0]], '6:1': [[1279224.0, 2621637.0, -7.0, -8.0], [3488026.0, 3431719.0, 4788283.0, 2107358.0], [1766671.0, 2030398.0, 1971471.0, 1971469.0], [1280687.0, 1745587.0, 1709642.0, 1044440.0]], '6:2': [[59524.0, 4273160.0, 2637119.0, -6.0], [3058774.0, 3431720.0, 4779373.0, 3920674.0], [1766671.0, 1895234.0, 3305536.0, 1005285.0], [1709641.0, 1823969.0, 1616754.0, 1600253.0]], '4:2': [[102987.0, 2619111.0, -7.0, 59526.0], [4788284.0, 4882876.0, 4853447.0, 4788284.0], [2030397.0, 3385725.0, 3292471.0, 3292473.0], [1624979.0, 1608618.0, 1624979.0, 1709643.0]], '7:1': [[-7.0, -7.0, 2619109.0, 8129787.0], [1151739.0, 1992669.0, -2.0, 2107359.0], [1971469.0, 1971468.0, 1971470.0, -4.0], [-6.0, 1709641.0, -5.0, -6.0]], '7:2': [[7807005.0, 8202804.0, 2637118.0, 11574235.0], [2048413.0, 3484998.0, 4460715.0, 2868892.0], [1971469.0, 80505.0, 1933153.0, 647580.0], [1703131.0, 1814659.0, 1729969.0, 1600252.0]], '7:3': [[8202805.0, 3089738.0, 2637119.0, 1353860.0], [3484999.0, 2828675.0, 3508227.0, 3309773.0], [1933152.0, 2628899.0, 1116577.0, 1202636.0], [1796340.0, 1778722.0, 1823969.0, 1823967.0]], '7:4': [[3089737.0, 1731832.0, 18555.0, 11762873.0], [3508226.0, 3309772.0, 4136410.0, 3309772.0], [1202637.0, 2754187.0, 1933155.0, 506507.0], [1796341.0, 1690619.0, 1631500.0, 1646532.0]], '7:5': [[8690726.0, 12583331.0, -128846.0, 4901530.0], [4119900.0, 3830100.0, -513149.0, 3193326.0], [2754186.0, 2787142.0, 114480.0, 2625954.0], [1709641.0, 1753601.0, -26255.0, 1619361.0]], '2:1': [[-7.0, -6.0, 2438209.0, 2388389.0], [4547085.0, 4547086.0, 1655567.0, 4680266.0], [1015692.0, 3283571.0, 3292468.0, 3292470.0], [1173222.0, 1462437.0, 1188130.0, 1314340.0]], '3:2': [[-6.0, -6873.0, 14040142.0, 48007.0], [4558604.0, 95796.0, 4788283.0, 4863157.0], [3283571.0, -127101.0, 3450219.0, 3299208.0], [1314340.0, -14973.0, 1462437.0, 1709642.0]], '1:1': [[13443781.0, 14210785.0, 2363782.0, 2438208.0], [-1.0, -2.0, -2.0, 4478959.0], [3292468.0, 3498483.0, 1294298.0, 3292469.0], [1314338.0, 1446918.0, 1390945.0, 1316457.0]], '2:2': [[-7.0, 14779700.0, 14821125.0, 1000.0], [3309773.0, 3706976.0, 3706976.0, 4788284.0], [1378053.0, 3307943.0, 3498483.0, 3299207.0], [1336238.0, 1497174.0, 1462436.0, 1624979.0]], '1:2': [[14124349.0, 14821126.0, 14289684.0, 14625789.0], [-2.0, 3537705.0, 3537706.0, 3706977.0], [3450219.0, 3590583.0, 3498483.0, 3498482.0], [1390945.0, 1462437.0, 1462436.0, 1462437.0]], '2:3': [[14779699.0, 14821128.0, 14821126.0, -77301.0], [4414617.0, 0, 4414615.0, 11436.0], [3450219.0, 3609103.0, 3491175.0, -74874.0], [1497173.0, 1532456.0, 1462437.0, -78829.0]], '1:3': [[14821125.0, 14821127.0, 14821126.0, 14821127.0], [3537706.0, -2.0, 4414615.0, 4414616.0], [3498483.0, 3599793.0, 3599792.0, 3563557.0], [1462436.0, 1462438.0, 1462437.0, 1497174.0]], '1:4': [[14821126.0, 14821128.0, 14821127.0, 14739474.0], [-2.0, -2.0, -2.0, 4293691.0], [3599792.0, 3529520.0, 3505894.0, 3609103.0], [1462437.0, 1479057.0, 1462436.0, 1462439.0]], '1:5': [[14821127.0, 14700148.0, 14700147.0, 14863756.0], [-1.0, -2.0, -1.0, 3754330.0], [3572466.0, 3293978.0, 3299206.0, 3443811.0], [1479056.0, 1557545.0, 1424594.0, 1535575.0]], '1:6': [[14637702.0, 14726267.0, 14674830.0, 14674831.0], [260410.0, 3780552.0, 3746017.0, 3790265.0], [3293979.0, 3186604.0, 3293978.0, 3096845.0], [1535574.0, 1531147.0, 1557545.0, 1608617.0]], '2:6': [[14821129.0, 14674832.0, 14674830.0, 14793313.0], [3861729.0, 4043451.0, -3.0, 3809980.0], [2553107.0, 3173394.0, 3134667.0, 3563559.0], [1659262.0, 1554338.0, 1608616.0, 1557547.0]], '3:6': [[14863757.0, 14907286.0, 14807119.0, 14821128.0], [3714586.0, 4293687.0, 260410.0, 3762738.0], [3563560.0, 3563558.0, 3554749.0, 3590587.0], [1787434.0, 1575691.0, 1575691.0, 1716253.0]], '4:6': [[14821129.0, 14687441.0, 14892676.0, 14821127.0], [4636113.0, 4293688.0, 3861725.0, 4351517.0], [3581479.0, 3401948.0, 3269165.0, 3618615.0], [1823971.0, 1564367.0, 1716252.0, 1852700.0]], '4:5': [[14482874.0, 14210790.0, 14821130.0, 14139680.0], [4703695.0, 4410743.0, 4293691.0, 4425489.0], [3637932.0, 3599796.0, 3554751.0, 3657652.0], [1716253.0, 1823970.0, 1823970.0, 1833381.0]], '5:5': [[6637899.0, 14306006.0, 2892453.0, -74428.0], [4703694.0, 4351517.0, 4410744.0, -171856.0], [3572468.0, 3590588.0, 3637933.0, 3667662.0], [1787435.0, 1842991.0, 1823971.0, -292968.0]], '5:4': [[5638692.0, 14266560.0, 2219230.0, 137633.0], [4695784.0, 4425489.0, 4703695.0, 4410745.0], [3521415.0, 3521417.0, 3590587.0, 3419180.0], [1778725.0, 1842990.0, 1823970.0, 1745589.0]], '6:4': [[2618184.0, -8964.0, 18556.0, -4.0], [4043455.0, -370448.0, 4695785.0, 3908565.0], [3419179.0, 68308.0, 3581477.0, 2625954.0], [1668880.0, -42501.0, 1753599.0, 1614445.0]], '6:3': [[2637118.0, -5.0, 5638692.0, -5.0], [4014538.0, 4425489.0, 4186837.0, 3309774.0], [2511683.0, 3470052.0, 3301219.0, 1933151.0], [1745587.0, 1663972.0, 1833379.0, 1814659.0]], '8:3': [[10316083.0, 8093481.0, 8202804.0, 12682844.0], [3148009.0, 2789656.0, 3484998.0, 3058773.0], [773232.0, 1933153.0, 773234.0, 29331.0], [1787430.0, 1745585.0, 1823968.0, 1823966.0]], '9:3': [[12682843.0, 12682845.0, 12241562.0, 12387389.0], [1992669.0, 2770546.0, 3484997.0, 1558402.0], [-4.0, 29330.0, 56159.0, -5.0], [1787429.0, 1745584.0, 1823967.0, 1716248.0]], '5:3': [[4273160.0, 6421036.0, 2619720.0, 5638691.0], [4779373.0, 4695785.0, 4602070.0, 4410744.0], [3456733.0, 3590586.0, 3463342.0, 3419179.0], [1770115.0, 1833380.0, 1709643.0, 1770115.0]], '4:4': [[2619111.0, 14297797.0, 14482875.0, 9726864.0], [4806404.0, 4703694.0, 4410746.0, 4703694.0], [3572466.0, 3637933.0, 3590586.0, 3590586.0], [1709643.0, 1823971.0, 1787433.0, 1616756.0]], '3:4': [[-34364.0, 14849448.0, 14821128.0, 14394593.0], [431940.0, 3851021.0, 4000229.0, 4410747.0], [-158238.0, 3628223.0, 3572467.0, 3590587.0], [-98647.0, 1787434.0, 1532456.0, 1535576.0]], '8:2': [[10975117.0, 12241562.0, 8202805.0, 12338680.0], [2107359.0, 2107361.0, 3920674.0, 1714791.0], [-4.0, 1202636.0, 3705.0, -4.0], [1600251.0, 1787431.0, 1745586.0, 1729966.0]], '9:2': [[12436197.0, 12682844.0, 12338679.0, 11480517.0], [1573835.0, 1606291.0, 2828674.0, 1584155.0], [-4.0, -4.0, -4.0, 443863.0], [1601568.0, 1805448.0, 1787430.0, 1716247.0]], '10:2': [[12485105.0, 12583325.0, 9892794.0, 11480517.0], [2789653.0, 3193326.0, 1584156.0, 1589464.0], [-6.0, 501499.0, -5.0, 443863.0], [990603.0, 1614441.0, 1787429.0, 1614440.0]], '10:1': [[12338678.0, 12485106.0, 12338679.0, 11857788.0], [1544564.0, 2789654.0, 1578944.0, 2789653.0], [154742.0, 478453.0, 154741.0, 438944.0], [990603.0, 1678993.0, -7.0, 665597.0]], '10:3': [[12583324.0, 12338681.0, 12682844.0, 12583325.0], [-3.0, -3.0, 3309772.0, 1589465.0], [213297.0, 501500.0, 12013.0, 443864.0], [1614440.0, 1614440.0, 1716249.0, 1659258.0]], '10:4': [[4370201.0, 9768169.0, 12682845.0, 9644641.0], [1766208.0, 2332808.0, 3148008.0, 2195517.0], [438946.0, 511619.0, -3.0, 154745.0], [938969.0, 1608614.0, 1745584.0, 591883.0]], '10:5': [[10840793.0, 9768170.0, 12096838.0, 7631465.0], [2195517.0, 3193323.0, 3193323.0, 3193322.0], [478455.0, 516829.0, -3.0, 478456.0], [1379029.0, 1143707.0, 1614442.0, 591884.0]], '10:6': [[5092000.0, 9768171.0, 11621249.0, 8387955.0], [3193322.0, -4.0, 3309769.0, 2332809.0], [-2.0, -2.0, 2567116.0, 2446436.0], [1072061.0, 1003918.0, 1551231.0, 154863.0]], '9:6': [[12241565.0, 12338683.0, 12832875.0, 11574239.0], [3193323.0, 3193323.0, 3309770.0, 3193323.0], [-2.0, 2567117.0, 2446438.0, 2446436.0], [1607006.0, 1542703.0, 1551232.0, 1143707.0]], '9:7': [[12338684.0, 9320172.0, 12338684.0, 10840796.0], [3193324.0, 2732824.0, 3193324.0, 3193322.0], [2567116.0, 2525389.0, 2625954.0, 2446433.0], [1003918.0, 1470449.0, 1551231.0, 1114590.0]], '8:7': [[12338685.0, 10619754.0, 10840799.0, 11715566.0], [3309770.0, 3148005.0, 3309768.0, 3193323.0], [2625955.0, 2625953.0, 2498270.0, 2567117.0], [1601570.0, 1339452.0, 1410580.0, 1470450.0]], '9:5': [[12682845.0, 12782765.0, 12338684.0, 12096837.0], [1612101.0, 3193324.0, 2789655.0, 3193322.0], [-2.0, 522139.0, -2.0, -2.0], [1608614.0, 1607005.0, 1650640.0, 1379030.0]], '9:4': [[12338681.0, 12782764.0, 11294883.0, 12338681.0], [2993846.0, 2732825.0, 3193327.0, 3148007.0], [29331.0, -2.0, 1202637.0, 511618.0], [1770112.0, 1619360.0, 1594525.0, 1696820.0]], '8:4': [[11248773.0, 8690728.0, 8021162.0, 11810283.0], [3262655.0, 3309771.0, 4088082.0, 3193326.0], [773233.0, 20625.0, 2244266.0, 511617.0], [1805449.0, 1690618.0, 1631499.0, 1703130.0]], '8:5': [[10840793.0, 12534121.0, 9320171.0, 10975122.0], [3309772.0, 3193325.0, 3148007.0, 3193323.0], [2141339.0, 2625955.0, 2262985.0, -3.0], [1616753.0, 1601570.0, 1753600.0, 1601570.0]], '8:6': [[12000921.0, 12338684.0, 12883085.0, 12732756.0], [3309771.0, 3193324.0, 3309771.0, 3309769.0], [2525391.0, 2625954.0, 2628901.0, 2567116.0], [1616752.0, 1470451.0, 1551233.0, 1601569.0]], '8:8': [[10445417.0, 10402208.0, 11527335.0, 9481608.0], [3309769.0, 2951123.0, 2951123.0, 3193322.0], [2625954.0, 2625952.0, 2642553.0, 2567116.0], [1470451.0, -1.0, 1003918.0, 134448.0]], '7:8': [[12883086.0, 10751981.0, 12883088.0, 10751981.0], [2789654.0, 2828668.0, 2808959.0, 3193323.0], [2459245.0, 2642552.0, 2658699.0, 2625953.0], [1410580.0, 951383.0, 938974.0, 1003919.0]], '6:8': [[-1580797.0, 11248781.0, 14259154.0, 12883087.0], [-102237.0, 3193320.0, 4000226.0, 2808960.0], [-91192.0, 2688095.0, 2925202.0, 2642553.0], [65263.0, 938973.0, 1354387.0, 1379034.0]], '6:9': [[12883088.0, 9603736.0, 8962796.0, 10102846.0], [4000225.0, 1971544.0, 2828664.0, 3193321.0], [2567115.0, 2639841.0, 2812575.0, 2642552.0], [730785.0, -1.0, -2.0, 951383.0]], '5:9': [[8962797.0, 4971261.0, 14281781.0, 12387397.0], [4275179.0, 3193318.0, 2290483.0, 1971545.0], [2925202.0, 2759293.0, 2848121.0, 2754184.0], [1535577.0, 915053.0, 870017.0, 859310.0]], '5:8': [[14097001.0, 12832879.0, 14394599.0, 14139682.0], [4275180.0, 2889191.0, 156065.0, 3193319.0], [2871851.0, 2848120.0, 3186610.0, 2925201.0], [1709646.0, 195504.0, 848698.0, 1354388.0]], '5:10': [[12933497.0, -2.0, 4971262.0, 6421035.0], [1870845.0, 1870844.0, 2290482.0, 3193319.0], [2848120.0, -3.0, 2675533.0, 2639841.0], [938972.0, 848699.0, -3.0, 747299.0]], '4:3': [[-7.0, -5.0, -7964.0, 5638692.0], [4853448.0, 4806403.0, 4892886.0, 4695784.0], [3325098.0, 3396444.0, -209693.0, 3590585.0], [1624980.0, 1608619.0, -31055.0, 1745589.0]], '5:6': [[9726865.0, 14349052.0, 14821128.0, 413907.0], [4425489.0, 4275180.0, 4221553.0, 3309770.0], [3637934.0, 3572467.0, 3572469.0, 3072529.0], [1842990.0, 1823972.0, 1823970.0, 1852701.0]], '5:7': [[12984005.0, 12883089.0, 14878167.0, -114401.0], [4293690.0, 3986017.0, 4275179.0, -183792.0], [3572468.0, 3004974.0, 2880359.0, -138902.0], [1833382.0, 1709645.0, 1709645.0, 286869.0]], '4:7': [[14821128.0, 14625797.0, 14907286.0, 14687440.0], [4636112.0, 4293687.0, 4293687.0, 4293689.0], [3581478.0, 3096847.0, 3554749.0, 3096847.0], [-3.0, 1526913.0, 1497177.0, 1770118.0]], '3:7': [[14892676.0, 14921996.0, 14892676.0, 14878167.0], [3762737.0, 4256968.0, 4204041.0, 4410742.0], [3563559.0, 3269163.0, 3269163.0, 3554748.0], [1642626.0, 915051.0, 1527821.0, 1497178.0]], '3:8': [[14835241.0, 14849450.0, 14937006.0, 14863759.0], [4275178.0, 4119895.0, -106110.0, 4275178.0], [3554749.0, 3186608.0, -134220.0, 3186610.0], [1557546.0, -2.0, -3882.0, 1526913.0]], '4:8': [[14579960.0, 14259154.0, 14892678.0, 14394598.0], [4410742.0, 4256968.0, 4275177.0, 4136407.0], [3450220.0, 3037499.0, 3186609.0, 2925202.0], [1580103.0, 938971.0, -2.0, 1354387.0]], '4:9': [[14849450.0, 11202773.0, 14452553.0, 12933497.0], [4256969.0, 4000223.0, 4221549.0, 4275178.0], [2973948.0, 3037498.0, 3173399.0, 2840713.0], [870016.0, 790544.0, 828380.0, 1354386.0]], '3:9': [[14863760.0, 14625794.0, 14674832.0, 14452552.0], [4275177.0, 4119892.0, 4275175.0, 4256968.0], [3240845.0, 3173398.0, 3186607.0, 3173398.0], [1532457.0, 755707.0, 1526910.0, 938971.0]], '4:10': [[14452552.0, 6892174.0, -2.0, 6859964.0], [4221550.0, 3986014.0, 3958295.0, 3193318.0], [3173398.0, 3060516.0, 2730739.0, 2651478.0], [915053.0, -4.0, -5.0, 37659.0]], '3:5': [[14821129.0, 14892676.0, 14863756.0, 14821129.0], [4293692.0, 4153014.0, 3598023.0, 0], [3628222.0, 3563559.0, 3599795.0, 3647742.0], [1753597.0, 1787433.0, 1659262.0, 1823971.0]], '9:1': [[12000916.0, 12682843.0, 11248771.0, 11294880.0], [1573835.0, -4.0, 1573834.0, 1793051.0], [-5.0, -5.0, -5.0, 154742.0], [1601568.0, -6.0, 1616750.0, 990603.0]], '8:1': [[10975117.0, 10316083.0, 3132371.0, 11621244.0], [1416475.0, 2107360.0, 2107358.0, 1589462.0], [-4.0, -4.0, -4.0, -4.0], [1600251.0, 1709640.0, 1058149.0, 1600250.0]], '2:4': [[14821127.0, 14821129.0, 14739473.0, 14821129.0], [3664822.0, 260411.0, -2.0, 4410746.0], [3590584.0, 3505892.0, 3599793.0, 3609104.0], [1532455.0, 1642626.0, 1462438.0, 1462440.0]], '2:7': [[14674831.0, 347252.0, 14766194.0, 14907286.0], [3861728.0, -121983.0, 4043450.0, 4293687.0], [-3.0, -27773.0, 3186604.0, 3450221.0], [1557546.0, -41128.0, 509224.0, 1497177.0]], '2:5': [[14821128.0, 14625795.0, 14637702.0, 14863757.0], [3861730.0, -2.0, -1.0, -1.0], [3431296.0, 3269164.0, 3521411.0, 3628223.0], [1431903.0, 1479059.0, 1554337.0, 1745589.0]], '10:7': [[9768170.0, 5067492.0, 11668357.0, 244349.0], [3193323.0, 3058767.0, 3193323.0, 3058766.0], [527547.0, 2567115.0, 2375180.0, 2070420.0], [-2.0, -3.0, 1379033.0, -2.0]], '9:8': [[10840797.0, 4833404.0, 8845467.0, 4531276.0], [3193323.0, 3193321.0, 3148005.0, 3193321.0], [2567117.0, 2553106.0, 2596235.0, 2567115.0], [1470450.0, 659193.0, 207912.0, 601801.0]], '10:8': [[-4.0, 6240280.0, 3038697.0, 7926.0], [3193322.0, 2951121.0, 3193322.0, 3193321.0], [2386588.0, 2410006.0, 2567116.0, 2410007.0], [-2.0, 640565.0, 591886.0, 601801.0]], '9:9': [[9851087.0, 2980741.0, 6240282.0, 3535064.0], [3193322.0, 3193320.0, 3193322.0, 2828667.0], [2446435.0, 2446433.0, 2567116.0, 2498269.0], [659194.0, 56256.0, 607013.0, 652883.0]], '8:9': [[10445418.0, 5216548.0, 8690729.0, 5216548.0], [3309768.0, 3015349.0, 2828668.0, 3015349.0], [2625953.0, 2567115.0, 2459243.0, 2511679.0], [1003919.0, -2.0, 596696.0, -1.0]], '7:9': [[12000925.0, 8614208.0, 10102845.0, 9200147.0], [3193322.0, 2828665.0, 3193320.0, 3309767.0], [2654989.0, 2596232.0, 2637233.0, 2625952.0], [1379034.0, 678921.0, 9559.0, 634558.0]], '6:10': [[9603737.0, 3478731.0, 6421034.0, 4123090.0], [3193320.0, 1870847.0, 143517.0, -6.0], [2386590.0, 2567113.0, 2675532.0, 2639842.0], [108421.0, -2.0, 764218.0, 596695.0]], '7:10': [[10102846.0, 290594.0, 8614209.0, 290595.0], [2025691.0, 1895573.0, 1870847.0, 3058767.0], [2654988.0, 2498269.0, 2567113.0, 2567115.0], [678922.0, 596695.0, 596694.0, -2.0]], '10:9': [[3028787.0, 3058812.0, 6892168.0, 2999659.0], [3193321.0, 3193319.0, 3193321.0, 3148002.0], [2446434.0, 2485159.0, 2567115.0, 2498269.0], [92902.0, 634555.0, 659193.0, 634556.0]], '10:10': [[3058813.0, 3058812.0, 2935196.0, 2962222.0], [3193320.0, 3193319.0, 3193320.0, 3193319.0], [2511678.0, 2511677.0, 2553105.0, 2498268.0], [646674.0, 634555.0, 523304.0, 634555.0]], '9:10': [[2776408.0, 2776407.0, 3638028.0, 2876332.0], [3193321.0, 3015348.0, 2828666.0, 2951120.0], [2498270.0, 2511678.0, 2567115.0, 2291806.0], [569840.0, 56256.0, -2.0, -2.0]], '8:10': [[8614212.0, 2999660.0, 290594.0, 2776407.0], [3193322.0, 2828666.0, 2553332.0, 2828667.0], [2625952.0, 2567115.0, 2567114.0, 2553105.0], [12168.0, -2.0, -2.0, -2.0]], '7:6': [[11480523.0, 12883086.0, 12096840.0, 12782766.0], [3830101.0, 3309770.0, 2808963.0, 3309770.0], [2625955.0, 2625955.0, 3240846.0, 2581628.0], [1316460.0, 1470452.0, 1796347.0, 1551232.0]], '6:6': [[-3382.0, -49064.0, 10840800.0, 12338686.0], [-69628.0, -136392.0, 4275181.0, 3309771.0], [375058.0, -143420.0, 3240847.0, 2754188.0], [-246149.0, 1852702.0, 1852700.0, 1737683.0]], '1:7': [[14726266.0, 14452550.0, 13965877.0, 14807122.0], [3780551.0, 4043449.0, 3840406.0, 4043451.0], [2954029.0, 3186605.0, 3096841.0, 3173394.0], [1557545.0, 1554336.0, 1554337.0, 1554338.0]], '1:8': [[3222449.0, 14452551.0, 76425.0, 167562.0], [4043450.0, 3840404.0, 4043449.0, -145965.0], [3096841.0, 3186606.0, 3173396.0, -167484.0], [1554337.0, 1548217.0, 1554336.0, -23874.0]], '7:7': [[12682849.0, 12883087.0, -907848.0, 12338684.0], [3309771.0, 2361728.0, -449458.0, 3309769.0], [2625956.0, 2498271.0, -197739.0, 2459246.0], [1564369.0, 1410579.0, 380386.0, 1470451.0]], '2:9': [[156672.0, 14452551.0, 14452551.0, 14674833.0], [-294920.0, 4275174.0, 4275174.0, 4275176.0], [-189312.0, 3186606.0, 3186606.0, 3240844.0], [-74319.0, 915050.0, 938968.0, 1526911.0]], '2:10': [[14662323.0, 13289360.0, 3143287.0, 12984005.0], [4275175.0, 4221547.0, 4221546.0, 4119892.0], [3186607.0, 3134670.0, 3173396.0, 3121962.0], [1354383.0, -6.0, 938967.0, 915051.0]], '3:10': [[14849450.0, 496473.0, 14625795.0, 11202773.0], [4256967.0, 4058156.0, 4119893.0, 4000223.0], [3173399.0, 2826192.0, 3186606.0, 3060516.0], [1354384.0, 915051.0, 870014.0, 790544.0]], '1:10': [[3176615.0, 3222452.0, 12984005.0, 14452551.0], [4275174.0, 4256964.0, 4275173.0, 4221547.0], [3060516.0, 3173396.0, 2759291.0, 3186606.0], [848696.0, 938967.0, 938967.0, 938968.0]], '1:9': [[11433815.0, 13238150.0, 12984004.0, 14452552.0], [3840405.0, 4275173.0, 4119893.0, 4275175.0], [3186605.0, 3173396.0, 3173397.0, 3186607.0], [1548218.0, 938967.0, 938968.0, 1354383.0]]}
        self.canvas = None
        self.root = None
        self.upsilon = 0.5
        self.quit = True
        self.startPos = [  (10.5, 10.5),(10.5, 1.5), (1.5, 10.5),(1.5, 1.5)]
        #
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

    def run(self, agent_host, world_x, world_y):
        """run the agent on the world"""
        self.start = time.time()
        total_reward = 0

        self.prev_s = None
        self.prev_a = None

        self.goalIndex = self.getGoalNo()
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
    length = [3, 6, 6]
    width = [3, 5, 7]
    for x, z, block in zip(length, width, block_list):
        # my_mission.drawItem(length_random, 227, width_random, item_list[i])
        for j in range(item_no):
            my_mission.drawBlock(x, 226, z, block)


def setReward(my_mission, agent):
    reward_list = {0: '2.5,8.5', 1: '3.5,3.5', 2: '6.5,5.5', 3: '6.5,7.5'}
    block_list = [(2.5, 8.5), (3.5, 3.5), (6.5, 5.5), (6.5, 7.5)]
    num = reward_list.get(agent.goalIndex)
    x, z = num.split(',')
    x, z = float(x), float(z)
    my_mission.rewardForReachingPosition(x, 227.0, z, 100.0, 0.0)
    print((x, z), 'reward:', 100)
    for str in block_list:
        if not str == (x, z):
            x_, z_ = str
            print((x_, z_), 'reward:', -100)
            my_mission.rewardForReachingPosition(x_, 227.0, z_, -100.0, 0.0)


def setQuitBlock(my_mission):
    block_list = [(2.5, 8.5), (3.5, 3.5), (6.5, 5.5), (6.5, 7.5)]
    for x, z in block_list:
        print(x, z)
        # position of the agent, not the block
        my_mission.endAt(x, 227.0, z, 0.5)


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
release_resource(resource_no, length, width)
setQuitBlock(my_mission)
# Attempt to start a mission
max_retries = 3
cumulative_rewards = []
total_repeat = 1000

# start learning
for i in range(total_repeat):
    setReward(my_mission, agent)
    print(len(agent.startPos))
    if i % 50 == 0 and i != 0:
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
