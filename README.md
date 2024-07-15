# MA_buff

能量机关模型缓冲输出结构：
（8400x23）
box[cx, cy, w, h] + Score[R_ON,R_OFF,B_ON,B_OFF] + [5,3] keypoints
前四个为框的坐标和宽高，后面四个为四个类别的框（红亮，红灭，蓝亮，蓝灭）的置信度，在后面为5个点的x,y,s(s为点的置信度)
