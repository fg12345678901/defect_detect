训练出的模型（一个分类器和四个对应类别的分割器）都在logs目录下（太大了上传不上来）

对应的路径分别为：

logs/Classify/best_model.pth（识别下面四个类别）

logs/magnetic_model/best_model.pth（加背景“0”有6个类别（每个像素只会属于四个类别之一，0是背景类别）

logs/phone_model_final/best_model.pth（加背景4类）

logs/solar-panel_model/best_model.pth（加背景29类）

logs/steel_model/best_model_v11.pth（加背景5类）

（最后这个名字不太一样，注意一下）

分别对应['magnetic', 'phone', 'solar-panel', 'steel'] 这四个task

模型的训练都是用engine里的train_seg完成的（给对应的数据和输出头的类别数）然后train_seg会有val的环节，所以参考你写推理时可以参考这里面（train_seg）（或者参考同一目录下的infer.py也可以（需要提供task名和类别数））

然后分类器的推理方法有个nfer_cls.py你可以参考（还是在engine里）