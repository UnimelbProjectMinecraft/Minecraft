首先，
==========

利用q-learnig学习整个地图，并且记录下Goal以及各个reward得分的地点（左边？action？），分值。
  （也许，不同的reward成不同的goal，每个goal分别用一个表来记录完成的policy？）
  
然后，
==========

判断(前期if else，后期可转神经网络)当前决定哪个goal，然后决定用哪个table值，完成选定goal。

每完成一个goal，
==========
进行下一次goal的选取判断（已完成的goal从选择list中删除），直至最后完成最大的goal。
