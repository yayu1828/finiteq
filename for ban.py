import numpy as np
import matplotlib.pyplot as plt
import csv
from multiprocessing import Pool, cpu_count
import time
import math
start_time = time.time()
def integrate_2d(func, x0, x1, y0, y1, dx, dy, *args):
    nx = int(np.ceil((x1 - x0) / dx))
    ny = int(np.ceil((y1 - y0) / dy))
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    result = 0.0
    for i in range(nx - 1):
        for j in range(ny - 1):
            f0 = func(x[i], y[j], *args)
            f1 = func(x[i + 1], y[j], *args)
            f2 = func(x[i], y[j + 1], *args)
            f3 = func(x[i + 1], y[j + 1], *args)
            if np.any(np.isinf([f0, f1, f2, f3])) or np.any(np.isnan([f0, f1, f2, f3])):
                continue
            average = (f0 + f1 + f2 + f3) / 4
            result += average * (x[i + 1] - x[i]) * (y[j + 1] - y[j])
    return result

n = 20
E = 0.001
A = 2.625
J = 1.325
m = 2
mass = 0.01575
Cross = 0.001519

def g(x, y, T, q, B, mass, Cross, m, n):
    numerator1 = (2*x**2/mass + 2*y**2/mass + q**2/(2*mass) - m - 2*n - Cross*2*q*B/mass + (Cross**2)*2*B**2/mass)
    energy1 = (-x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) - x*q/(mass*T) + Cross*x*B/(mass*T) + Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + m/(2*T) + n/T) * 11.6
    energy2 = (x*q/(mass*T) - x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) - Cross*x*B/(mass*T) + Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + n/T + m/(2*T)) * 11.6
    numerator2 = np.exp(energy1) / (np.exp(energy1) + 1)
    numerator3 = 1 / (np.exp(energy2) + 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2 / numerator1
    elif energy1 < -700:
        return -numerator3 / numerator1
    else:
        return (numerator2 - numerator3) / numerator1

def f(x, y, T, q, B, mass, Cross, m, n):
    numerator1 = (2*x**2/mass + 2*y**2/mass + q**2/(2*mass) - 2*n - Cross*2*x*B/mass + (Cross**2)*2*B**2/mass)
    energy1 = (-x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) - x*q/(mass*T) + Cross*x*B/(mass*T) + Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + m/(2*T) + n/T) * 11.6
    energy2 = (x*q/(mass*T) - x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) + Cross*x*B/(mass*T) - Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + n/T - m/(2*T)) * 11.6
    numerator2 = np.exp(energy1) / (np.exp(energy1) + 1)
    numerator3 = 1 / (np.exp(energy2) + 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2 / numerator1
    elif energy1 < -700:
        return -numerator3 / numerator1
    else:
        return (numerator2 - numerator3) / numerator1

def F(x, y, T, q, B, mass, Cross, m, n):
    numerator1 = (2*x**2/mass + 2*y**2/mass + q**2/(2*mass) - 2*n + Cross*2*x*B/mass + (Cross**2)*2*B**2/mass)
    energy1 = (-x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) - x*q/(mass*T) - Cross*x*B/(mass*T) - Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + n/T - m/(2*T)) * 11.6
    energy2 = (x*q/(mass*T) - x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) - Cross*x*B/(mass*T) + Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + n/T + m/(2*T)) * 11.6
    numerator2 = np.exp(energy1) / (np.exp(energy1) + 1)
    numerator3 = 1 / (np.exp(energy2) + 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2 / numerator1
    elif energy1 < -700:
        return -numerator3 / numerator1
    else:
        return (numerator2 - numerator3) / numerator1

def h(x, y, T, q, B, mass, Cross, m, n):
    numerator1 = (2*x**2/mass + 2*y**2/mass + q**2/(2*mass) - 2*n + m + Cross*2*q*B/mass + (Cross**2)*2*B**2/mass)
    energy1 = (-x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) - x*q/(mass*T) - Cross*x*B/(mass*T) - Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + n/T - m/(2*T)) * 11.6
    energy2 = (x*q/(mass*T) - x**2/(mass*T) - y**2/(mass*T) - q**2/(4*mass*T) + Cross*x*B/(mass*T) - Cross*q*B/(2*mass*T) - (Cross**2)*B**2/(mass*T) + n/T - m/(2*T)) * 11.6
    numerator2 = np.exp(energy1) / (np.exp(energy1) + 1)
    numerator3 = 1 / (np.exp(energy2) + 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2 / numerator1
    elif energy1 < -700:
        return -numerator3 / numerator1
    else:
        return (numerator2 - numerator3) / numerator1

matrix_v = np.full((4, 4), E)
matrix_v[0, 0] = A
matrix_v[3, 3] = A
matrix_v[0, 3] = J
matrix_v[3, 0] = J
matrix_V = -matrix_v

def matrix(T, q, B, mass, Cross, m, n):
    CC1 = integrate_2d(g, -2, 2, -2, 2, 0.04, 0.04, T, q, B, mass, Cross, m, n)
    CC2 = integrate_2d(f, -2, 2, -2, 2, 0.04, 0.04, T, q, B, mass, Cross, m, n)
    CC3 = integrate_2d(F, -2, 2, -2, 2, 0.04, 0.04, T, q, B, mass, Cross, m, n)
    CC4 = integrate_2d(h, -2, 2, -2, 2, 0.04, 0.04, T, q, B, mass, Cross, m, n)
    matrix_X = np.diag([CC1, CC2, CC3, CC4])
    product_matrix = np.dot(matrix_V, matrix_X)
    eigenvalues_product = np.linalg.eigvals(product_matrix)
    max_eigenvalue = np.max(eigenvalues_product)
    return max_eigenvalue

def binary_search_eigenvalue(function,q,B,mass,Cross,m,n, low, high, tolerance, target):
    if (function(low,q,B,mass,Cross,m,n) - target) * (function(high,q,B,mass,Cross,m,n) - target) > 0:
        return None
      # 计算需要迭代的次数
    iterations = int(math.ceil(math.log((high - low) / tolerance, 2)))
    for _ in range(iterations):
        mid = (low + high) / 2.0  # 计算中间点
        mid_value = function(mid, q, B, mass, Cross, m, n)  # 计算中间点的函数值
        # 如果当前的中点值已经足够接近目标值，则可以提前退出循环
        if abs(mid_value - target) < tolerance:
            return mid
        # 根据函数值更新搜索区间
        if mid_value < target:
            high = mid
        else:
            low = mid
        # 如果区间已经足够小，则退出循环
        if high - low <= tolerance:
            break
    # 返回最终的中间点
    return (low + high) / 2.0

def calculate_Tc(B, q_values, mass, Cross, m, n):
    T_values = []
    q_nonzero=[]
    for q in q_values:
        T = binary_search_eigenvalue(matrix, q, B, mass, Cross, m, n, 0.1, 100, 0.001, 1)
        if T is not None:  # 如果找到了结果，则添加到T_list列表中
            T_values.append(T)
            q_nonzero.append(q)
        else:
            print(f"在 B={B},q = {q} 时没有找到结果，已忽略此点。")
    Tc_max = max(T_values)
    max_index = np.argmax(T_values)
    return Tc_max, q_nonzero[max_index], T_values,q_nonzero

def calculate_Tc_wrapper(args):
    return calculate_Tc(*args)

if __name__ == "__main__":
    q_values = np.linspace(0, 0.04, 81)
    B_values = np.linspace(0, 20, 21)
    Tc = []
    q_real = []
    

    num_cores = cpu_count()
    print(f"Number of available CPU cores: {num_cores}")

    with Pool(processes=num_cores-1) as pool:
        results = pool.map(calculate_Tc_wrapper, [(B, q_values, mass, Cross, m, n) for B in B_values])
    
    # 收集所有数据
    all_data = []
    for result, B in zip(results, B_values):
        Tc_max, q_max, T_values,q_nonzero= result
         # 将Tc_max和q_max分别存入Tc和q_real数组中
        Tc.append(Tc_max)
        q_real.append(q_max)
        for q, T in zip(q_nonzero, T_values):
            all_data.append((B, q, T))

    # 一次性写入所有数据到 CSV 文件
    with open('q_nonzero_Tc_datafor2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['B', 'q', 'Tc'])
        for row in all_data:
            writer.writerow(row)

    # 绘制图形
    plt.figure()
    for result, B in zip(results, B_values):
        Tc_max, q_max, T_values,q_nonzero= result
        plt.plot(q_nonzero, T_values, 'o-', label=f'B = {B}')
    plt.legend()
    plt.xlabel('q/nm-1')
    plt.ylabel('Tc/K')
    plt.title('q-Tc')
    plt.grid(True)
    ax = plt.gca()
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('q_nonzero_Tc_datafor2.svg', format='svg')
    plt.show()

    filename = 'B-Tc-nonzerofor2.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['B', 'Tc', 'q_real'])
        for i in range(len(B_values)):
            writer.writerow([B_values[i], Tc[i], q_real[i]])

    plt.figure()
    plt.plot(Tc, B_values, 'o-')
    plt.xlabel('Tc/K')
    plt.ylabel('B/T')
    plt.title(f'B-Tc Diagram Δμ={m}')
    plt.grid(True)
    plt.savefig('B-Tc-nonzerofor2.svg', format='svg')
    plt.show()
    end_time = time.time()
    print(f"运行时间：{end_time - start_time}秒")