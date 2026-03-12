# test_import.py - 仅测试导入，不执行任何训练
print("1. 开始导入tensorflow...")
import tensorflow as tf
print("✅ tensorflow导入完成！")

print("2. 开始导入load_data...")
from load_data import load_wss_data, load_amc_data
print("✅ load_data导入完成！")

print("3. 开始导入build_models...")
from build_models import build_wssnet, build_amcnet, get_train_callbacks
print("✅ build_models导入完成！")

print("\n🎉 所有依赖导入正常！")