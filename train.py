import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据参数
no_sequences = 30
sequence_length = 30
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['ok', 'like', '1', '2', '3', '4', '5'])

# 标签映射
label_map = {label: num for num, label in enumerate(actions)}

# 数据预处理函数
def preprocess_data():
    # 加载数据
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    # 将数据转换为 NumPy 数组
    X = np.array(sequences)
    y = np.array(labels)

    # 数据归一化
    X = (X - np.mean(X)) / np.std(X)

    # 重塑数据为 4 维
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 增加测试集大小

    return X_train, X_test, y_train, y_test

# 数据增强生成器
def create_data_augmentation():
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# 定义模型
def build_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, activation='relu', input_shape=(sequence_length, 258)))  # 增加 LSTM 神经元数量
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    return model

# 训练模型
def train_model(X_train, y_train):
    model = build_model()

    # 编译模型
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 定义回调函数
    callbacks = [
        TensorBoard(log_dir='Logs'),
        EarlyStopping(monitor='val_loss', patience=20)  # 增加 patience
    ]

    # 训练模型
    model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_split=0.2, callbacks=callbacks)  # 调整训练轮数、批量大小和验证集大小

    return model

# 主函数
def main():
    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data()

    # 创建数据增强生成器
    datagen = create_data_augmentation()

    # 应用数据增强
    X_train = datagen.flow(X_train, y_train, batch_size=64).next()[0]

    # 训练模型
    model = train_model(X_train, y_train)

    # 保存模型
    model.save('action.h5')

if __name__ == '__main__':
    main()