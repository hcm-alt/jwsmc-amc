import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os

# ===================== WSSNet（无Reshape，无SAM函数调用） =====================
def build_wssnet(input_shape=(30, 128, 2)):
    """WSSNet - 极简版，彻底移除所有Reshape层"""
    inputs = layers.Input(shape=input_shape)
    
    # 卷积层（无维度错误）
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), strides=(1,2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 注意力模块（无Reshape，直接池化+卷积）
    max_pool = layers.MaxPooling2D(pool_size=(1,1), padding='same')(x)
    avg_pool = layers.AveragePooling2D(pool_size=(1,1), padding='same')(x)
    concat = layers.Concatenate(axis=-1)([max_pool, avg_pool])
    attention = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid')(concat)
    x = x * attention  # 直接加权，无Reshape
    
    # 分类头
    x = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(30, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="WSSNet")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model

# ===================== AMCNet（增强版 - 优化1：模型架构） =====================
def build_amcnet(input_shape=(256, 2)):
    """AMCNet - 增强版：更深网络 + 注意力机制 + 残差连接"""
    inputs = layers.Input(shape=input_shape)
    
    # ===== 特征提取阶段 =====
    # 第一层卷积块
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # 第二层卷积块
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # 第三层卷积块
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # ===== 注意力机制 =====
    # 通道注意力
    gap = layers.GlobalAveragePooling1D()(x)
    gap = layers.Dense(128, activation='relu')(gap)
    attention_weights = layers.Dense(x.shape[-1], activation='sigmoid')(gap)
    attention_weights = layers.Reshape((1, x.shape[-1]))(attention_weights)
    x = layers.Multiply()([x, attention_weights])
    
    # ===== LSTM特征融合 =====
    x = layers.LSTM(256, return_sequences=False, dropout=0.3)(x)
    x = layers.LayerNormalization()(x)
    
    # ===== 分类头 =====
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="AMCNet")
    
    # 使用学习率优化的Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ===================== AMCNet（原始极简版 - 保留对比） =====================
def build_amcnet_original(input_shape=(256, 2)):
    """AMCNet - 原始极简版"""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="AMCNet_original")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ===================== 回调函数 =====================
def get_train_callbacks(model_name, log_dir="../logs", model_dir="../model"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(f"{model_dir}/{model_name}_best.h5", monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=f"{log_dir}/{model_name}")
    ]

# ===================== 测试 =====================
if __name__ == "__main__":
    try:
        print("开始搭建WSSNet...")
        wss = build_wssnet()
        print("WSSNet搭建成功！")
        wss.summary()
        
        print("\n" + "-"*50 + "\n")
        
        print("开始搭建AMCNet...")
        amc = build_amcnet()
        print("AMCNet搭建成功！")
        amc.summary()
    except Exception as e:
        print(f"报错：{type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()