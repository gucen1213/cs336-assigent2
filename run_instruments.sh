#!/bin/bash

# 第一步：强制切换到你的工作目录
# Instruments 默认的工作路径可能不是你的代码文件夹，这极易导致脚本内部相对路径（如读取数据集）报错闪退
cd /Users/guweijie/Desktop/PythonProject/VsProject/CS336/assignment2-systems-main/cs336_systems/

# 第二步：在你的 Python 绝对路径前，加上 exec ！！！
# （注意替换为你真实的 Python 绝对路径，这里我写的是示例）
exec /Users/guweijie/Desktop/PythonProject/VsProject/CS336/assignment2-systems-main/.venv/bin/python memory_script.py