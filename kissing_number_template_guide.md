# Kissing Number模板使用指南

## 概述

我已经为你的11维kissing number问题创建了专门的模板，这些模板针对几何优化问题进行了优化，与一般的代码优化模板有很大不同。

## 新增的Kissing Number专用模板

### 1. 系统模板

#### `kissing_number_system` - 专家系统角色
- 将AI定位为计算几何和优化专家
- 强调11维欧几里得空间的球体打包理论
- 专注于几何约束满足和球数量最大化

#### `kissing_number_evaluator_system` - 专家评估者角色
- 专门评估kissing number算法
- 关注数学正确性和几何优化能力

### 2. 用户提示模板

#### `kissing_number_diff_user` - 渐进式改进
**关键特性:**
- 显示当前球数量和约束满足率
- 明确593球的目标基准
- 强调数学约束的重要性:
  - 非退化性: 0 ∉ C
  - kissing条件: min||x-y|| ≥ max||x||
- 提供针对几何优化的改进建议

**参数:**
- `{sphere_count}`: 当前找到的球数量
- `{constraint_satisfaction}`: 约束满足率
- `{performance_gap}`: 与593球目标的差距
- `{improvement_areas}`: 具体改进建议

#### `kissing_number_full_rewrite` - 完全重新设计
**用于:**
- 算法架构完全重新设计
- 尝试全新的优化策略
- 集成多种metaheuristic方法

### 3. 评估模板

#### `kissing_number_evaluation` - 专业几何评估
**评估维度:**
1. **约束满足** (constraint_satisfaction): 几何约束遵守程度
2. **球数量性能** (sphere_count_performance): 球数量最大化效果
3. **数值稳定性** (numerical_stability): 11维计算的精度和稳定性
4. **搜索效率** (search_efficiency): 配置空间探索效率

## 使用方法

### 在代码中使用
```python
from evolve_agent.prompt.templates import TemplateManager

# 初始化模板管理器
template_manager = TemplateManager()

# 获取kissing number专用模板
system_template = template_manager.get_template("kissing_number_system")
user_template = template_manager.get_template("kissing_number_diff_user")
eval_template = template_manager.get_template("kissing_number_evaluation")

# 格式化模板
formatted_prompt = user_template.format(
    sphere_count=current_count,
    constraint_satisfaction="95%", 
    performance_gap=593 - current_count,
    improvement_areas="需要改进搜索策略",
    artifacts="",
    evolution_history="",
    language="python",
    current_program=your_algorithm_code
)
```

### 配置文件中指定
在你的配置文件中可以这样指定:
```yaml
templates:
  system_message: "kissing_number_system"
  diff_user: "kissing_number_diff_user" 
  evaluation: "kissing_number_evaluation"
  full_rewrite_user: "kissing_number_full_rewrite"
```

## 关键改进点

### 1. 数学导向 vs 代码导向
- **原模板**: 关注代码可读性、可维护性、一般性能
- **新模板**: 关注几何约束、球数量最大化、数值精度

### 2. 评估指标的转变
- **原指标**: readability, maintainability, efficiency
- **新指标**: constraint_satisfaction, sphere_count_performance, numerical_stability, search_efficiency

### 3. 优化建议的专业化
- **原建议**: 一般的代码优化技巧
- **新建议**: 几何构造方法、格子理论、高维优化策略

### 4. 约束意识
- 明确强调必须满足的数学约束
- 提供约束验证的重要性
- 平衡约束满足与性能优化

## 最佳实践

1. **始终使用专用模板**: 对于kissing number问题，使用`kissing_number_*`系列模板
2. **提供准确的度量**: 确保`sphere_count`和`constraint_satisfaction`参数准确
3. **强调数学验证**: 每次改进后都要验证几何约束
4. **渐进式优化**: 先用`diff_user`模板进行小幅改进，再考虑`full_rewrite`

这些专用模板将帮助你的进化算法更好地专注于kissing number问题的核心挑战，而不是分散注意力到一般的代码质量问题上。 