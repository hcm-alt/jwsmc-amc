import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger
from load_data import load_wss_data, load_amc_data
from build_models import build_wssnet, build_amcnet, get_train_callbacks


# ===================== 数据增强层（优化2：训练策略） =====================
class DataAugmentation(tf.keras.layers.Layer):
    """数据增强：添加随机噪声和相位偏移"""
    def __init__(self, noise_factor=0.1, **kwargs):
        super(DataAugmentation, self).__init__(**kwargs)
        self.noise_factor = noise_factor
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # 添加随机高斯噪声
        noise = tf.random.normal(shape=tf.shape(inputs), stddev=self.noise_factor)
        inputs = inputs + noise
        
        # 添加随机相位偏移
        phase = tf.random.uniform(shape=(tf.shape(inputs)[0], 1, 1), minval=0, maxval=2*3.14159)
        inputs = tf.cast(inputs, tf.complex64)
        inputs = inputs * tf.exp(1j * tf.cast(phase, tf.complex64))
        inputs = tf.math.real(inputs)
        
        return inputs


def train_wssnet():
    """训练WSSNet（频谱感知模型）"""
    print("="*60)
    print("开始训练WSSNet...")
    print("="*60)
    
    # 加载数据
    x_train, x_val, x_test, y_train, y_val, y_test = load_wss_data()
    
    # 搭建模型
    model = build_wssnet()
    
    # 获取回调函数
    callbacks = get_train_callbacks("WSSNet")
    
    # 训练模型
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 测试模型
    print("\n" + "="*60)
    print("WSSNet测试集评估...")
    print("="*60)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"WSSNet测试集损失：{test_loss:.4f}")
    print(f"WSSNet测试集准确率：{test_acc:.4f}")
    
    return model, history

def train_amcnet():
    """训练AMCNet（调制识别模型）- 优化版"""
    print("\n" + "="*60)
    print("开始训练AMCNet（优化版）...")
    print("="*60)
    
    # 加载数据
    x_train, x_val, x_test, y_train, y_val, y_test = load_amc_data()
    
    # 搭建模型
    model = build_amcnet()
    
    # 打印模型信息
    print(f"\n模型参数数量: {model.count_params():,}")
    
    # 优化回调函数（学习率调度 + 早停 + 模型保存）
    callbacks = [
        # 学习率调度：当验证损失不再下降时降低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        # 模型保存
        ModelCheckpoint(
            "../model/AMCNet_best.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard
        TensorBoard(log_dir="../logs/AMCNet")
    ]
    
    # 训练模型 - 使用优化后的参数
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=300,  # 增加训练轮数
        batch_size=64,  # 增大batch size
        callbacks=callbacks,
        verbose=1
    )
    
    # 测试模型
    print("\n" + "="*60)
    print("AMCNet测试集评估...")
    print("="*60)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"AMCNet测试集损失：{test_loss:.4f}")
    print(f"AMCNet测试集准确率：{test_acc:.4f}")
    
    return model, history


def train_amcnet_with_augmentation():
    """训练AMCNet（带数据增强）"""
    print("\n" + "="*60)
    print("开始训练AMCNet（带数据增强）...")
    print("="*60)
    
    # 加载数据
    x_train, x_val, x_test, y_train, y_val, y_test = load_amc_data()
    
    # 创建带数据增强的模型
    inputs = layers.Input(shape=(256, 2))
    
    # 数据增强层
    aug = DataAugmentation(noise_factor=0.05)(inputs, training=True)
    
    # 基础模型
    base_model = build_amcnet()
    
    # 连接
    outputs = base_model(aug)
    model = Model(inputs, outputs)
    
    # 重新编译（使用优化后的学习率）
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 优化回调函数
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            "../model/AMCNet_aug_best.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(log_dir="../logs/AMCNet_aug")
    ]
    
    # 训练
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=300,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # 测试
    print("\n" + "="*60)
    print("AMCNet（带增强）测试集评估...")
    print("="*60)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"AMCNet测试集损失：{test_loss:.4f}")
    print(f"AMCNet测试集准确率：{test_acc:.4f}")
    
    return model, history

if __name__ == "__main__":
    try:
        print("1. 开始导入依赖...")
        import tensorflow as tf
        from load_data import load_wss_data, load_amc_data
        from build_models import build_wssnet, build_amcnet, get_train_callbacks
        print("依赖导入完成！")
        
        print("2. 开始训练WSSNet...")
        wss_model, wss_history = train_wssnet()
        print("WSSNet训练完成！")
        
        print("3. 开始训练AMCNet...")
        amc_model, amc_history = train_amcnet()
        print("AMCNet训练完成！")
        
        print("\n" + "="*60)
        print("所有模型训练完成！")
        print(f"WSSNet最优模型保存路径：../model/WSSNet_best.h5")
        print(f"AMCNet最优模型保存路径：../model/AMCNet_best.h5")
        print("="*60)
    except Exception as e:
        print(f"报错类型：{type(e).__name__}")
        print(f"报错信息：{e}")
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈