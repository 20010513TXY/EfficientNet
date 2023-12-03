# EfficientNet
use a model named EfficientNet to predict the class of flowers
1、utils.py用于分割数据集和训练数据、验证数据。其中的read_split_data方法将花数据集分为训练集和验证集，也可根据自己需要，将数据集分为训练集、验证集和测试集；train_one_epoch方法表示一个epoch的训练过程，evaluate方法表示一个epoch的验证过程
2、my_dataset.py对分割后的数据进行处理（没太看懂操作）
3、model.py是EfficientNet的模型结构，模型有些复杂，需要认真看，多debug
4、train.py就是利用分割后的数据集，在模型上进行训练和验证
5、predict.py利用训练好的模型对单张花图像进行类别预测
6、efficientnetb0.pth是从网上下载的预训练模型

注：代码源于https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test9_efficientNet，
本人只是学习了这个代码，并在部分代码上进行了一些标注利于理解
