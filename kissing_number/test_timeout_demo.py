#!/usr/bin/env python3
"""
演示timeout是如何在评估过程中出现的

重要说明:
- kissing_number/evaluator.py 本身没有timeout机制
- timeout是在上层的 evolve_agent/evaluator.py 中实现的
- 当进化过程运行时，上层评估器会调用我们的evaluate函数，并设置timeout保护
"""

import time
import tempfile
import os
from evaluator import evaluate

def create_slow_program(sleep_seconds):
    """创建一个执行时间很长的程序"""
    code = f'''
import time
import numpy as np

def main():
    print("程序开始执行...")
    # 模拟非常慢的计算 - 例如无限循环或复杂计算
    time.sleep({sleep_seconds})  # 睡眠{sleep_seconds}秒
    
    # 返回一个简单的有效配置
    sphere_centers = np.array([
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=float)
    print("程序执行完成！")
    return sphere_centers
'''
    return code

def demonstrate_timeout_concept():
    """演示timeout概念和机制"""
    
    print("EvolveAgent中的Timeout机制演示")
    print("=" * 60)
    print()
    
    print("📋 Timeout机制说明:")
    print("━" * 40)
    
    # 查看配置
    import sys
    sys.path.append('..')
    
    try:
        from evolve_agent.config import Config
        config = Config.from_yaml("../configs/default_config.yaml")
        timeout_setting = config.evaluator.timeout
        print(f"✓ 系统配置的timeout: {timeout_setting} 秒")
    except Exception as e:
        print(f"⚠ 无法读取配置: {e}")
        timeout_setting = 300
        print(f"✓ 默认timeout设置: {timeout_setting} 秒")
    
    print()
    print("🔧 Timeout的工作层次:")
    print("  1. kissing_number/evaluator.py (我们的评估函数)")
    print("     └─ 纯函数，直接评估程序，无timeout保护")
    print("  2. evolve_agent/evaluator.py (上层评估器)")
    print("     └─ 包含timeout机制，调用我们的evaluate函数")
    print("  3. 进化过程中会使用上层评估器")
    print("     └─ asyncio.wait_for(evaluate_function(), timeout=300)")
    print()
    
    print("⏰ Timeout触发的场景:")
    print("  • 程序包含无限循环")
    print("  • 计算复杂度过高（如大规模优化）")
    print("  • 程序等待用户输入")
    print("  • 系统资源不足导致程序卡死")
    print("  • 网络请求等外部依赖超时")
    print()
    
    print("🛡️ Timeout保护的作用:")
    print("  • 防止单个程序占用过多计算资源")
    print("  • 确保进化过程能继续进行")
    print("  • 避免系统因程序错误而卡死")
    print("  • 提供公平的计算时间限制")
    print()
    
    # 演示正常执行
    print("📊 实际测试演示:")
    print("━" * 40)
    
    print("测试1: 正常执行的程序（快速完成）")
    fast_code = create_slow_program(0.5)  # 0.5秒
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(fast_code)
        fast_file = f.name
    
    try:
        start_time = time.time()
        result = evaluate(fast_file)
        end_time = time.time()
        
        print(f"  ⏱️  执行时间: {end_time - start_time:.2f}秒")
        print(f"  🔢 球数量: {result.get('num_spheres', 'N/A')}")
        print(f"  ✅ 正常完成，无timeout")
    except Exception as e:
        print(f"  ❌ 执行出错: {e}")
    finally:
        os.unlink(fast_file)
    
    print()
    print("⚠️  注意: 当前直接调用evaluate()函数没有timeout保护")
    print("   只有在EvolveAgent进化过程中，上层评估器才会应用timeout")
    print()
    
    print("🚀 在实际进化中的Timeout处理:")
    print("  当程序超时时，evolve_agent/evaluator.py会:")
    print("  • 强制终止程序执行")
    print("  • 返回 {'timeout': True, 'combined_score': 0.0}")
    print("  • 记录timeout事件到日志")
    print("  • 继续下一个进化迭代")
    print()
    
    print("📝 常见触发timeout的程序模式:")
    print("  1. while True: # 无限循环")
    print("  2. 递归深度过深")
    print("  3. 大规模数值计算没有收敛")
    print("  4. 文件I/O或网络请求")
    print("  5. 内存分配过大导致系统响应慢")
    print()
    
    print(f"⚙️  当前系统timeout设置: {timeout_setting}秒")
    print("   可以在配置文件中调整这个值")
    print("   • 较短的timeout: 快速淘汰问题程序")
    print("   • 较长的timeout: 允许复杂算法有更多计算时间")

if __name__ == "__main__":
    demonstrate_timeout_concept() 