import math

# 定义一个简单的线性函数
def y(x):
    return 3.5 - x

# 修改后的二分搜索函数
def binary_search_eigenvalue(function, low, high, tolerance, target):
    # 在开始二分搜索之前检查端点值
    if (function(low) - target) * (function(high) - target) > 0:
        return "没有找到结果，所有函数值都在目标值的一侧。"

    # 计算需要迭代的次数
    for _ in range(int(math.ceil(math.log((high - low) / tolerance, 2)))):
        mid = (low + high) / 2.0  # 计算中间点
        mid_value = function(mid)  # 计算中间点的函数值

        # 如果当前的中点值已经足够接近目标值，则可以提前退出循环
        if abs(mid_value - target) < tolerance:
            return mid

        # 根据函数值更新搜索区间
        if mid_value < target:
            high = mid
        else:
            low = mid

    # 返回最终的中间点
    return (low + high) / 2.0

# 调用二分搜索函数并打印结果
T = binary_search_eigenvalue(y, 0, 1, 0.001, 1)
print(T)
