import tensorflow as tf
import time

print("=" * 60)
print("TensorFlow GPU 快速测试")
print("=" * 60)

# 基本信息
print(f"TensorFlow 版本: {tf.__version__}")

# GPU 检测
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU 数量: {len(gpus)}")

if gpus:
    print("✅ 找到 GPU:")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
    
    # 测试 GPU 计算
    print("\n🚀 测试 GPU 计算...")
    try:
        with tf.device('/GPU:0'):
            # 创建两个矩阵
            size = 1000
            print(f"   计算 {size}x{size} 矩阵乘法...")
            
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            
            start = time.time()
            c = tf.matmul(a, b)
            elapsed = time.time() - start
            
            print(f"   ✅ 计算成功!")
            print(f"   耗时: {elapsed:.3f} 秒")
            print(f"   结果形状: {c.shape}")
            
    except Exception as e:
        print(f"   ❌ GPU 计算失败: {e}")
else:
    print("❌ 未找到 GPU")

print("\n" + "=" * 60)