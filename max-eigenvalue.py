import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
def integrate_2d(func, x0, x1, y0, y1, dx, dy, *args):
    # 计算x和y的积分点数
    nx = int(np.ceil((x1 - x0) / dx))
    ny = int(np.ceil((y1 - y0) / dy))
    
    # 生成x和y的积分点
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    
    # 初始化积分结果
    result = 0.0
    
    # 对于x的每个区间
    for i in range(nx - 1):
        # 对于y的每个区间
        for j in range(ny - 1):
            # 计算积分小区间的四个顶点
            f0 = func(x[i], y[j], *args)
            f1 = func(x[i + 1], y[j], *args)
            f2 = func(x[i], y[j + 1], *args)
            f3 = func(x[i + 1], y[j + 1], *args)
            
            # 计算小矩形区域的平均值
            average = (f0 + f1 + f2 + f3) / 4
            
            # 将平均值乘以小矩形面积并累加到总积分中
            result += average * (x[i + 1] - x[i]) * (y[j + 1] - y[j])
            
    return result
n = 20
E = 0.001
A = 2.625
J = 1.325#2.5倍
m = 2# Δμ的值
q=0.0385
B=4
mass=0.01575#即乘\hbar^{2}/2m=63.48meV/nm²\
Cross=0.001519
def g(x, y,T,q,B, mass,Cross,m,n):#x是kx，y是ky，e是有效质量，n是μ，m是Δμ
    numerator1 = (2*x**2/mass+2*y**2/mass + q**2/(2*mass) -m-2*n-Cross*2*q*B/mass+(Cross**2)*2*B**2/mass)
    energy1=(-x**2/(mass*T) -y**2/(mass*T)- q**2/(4*mass*T) - x*q/(mass*T)+Cross*x*B/(mass*T)+Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T) + m/(2*T)+n/T) *11.6
    energy2=(x*q/(mass*T) - x**2/(mass*T)-y**2/(mass*T) - q**2/(4*mass*T)-Cross*x*B/(mass*T)+Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T)+n/T+m/(2*T))*11.6
    numerator2 = np.exp(energy1)/ (np.exp(energy1)+ 1)
    numerator3 = 1 / (np.exp(energy2)+ 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2/numerator1
    elif energy1 < -700:
        return -numerator3/numerator1
    else:
        return (numerator2 - numerator3) / numerator1

def f(x, y,T,q,B, mass,Cross,m,n):
    numerator1 = (2*x**2/mass +2*y**2/mass+ q**2/(2*mass)-2*n-Cross*2*x*B/mass+(Cross**2)*2*B**2/mass)
    energy1=(-x**2/(mass*T) - y**2/(mass*T)-q**2/(4*mass*T) - x*q/(mass*T) +Cross*x*B/(mass*T)+Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T) + m/(2*T)+n/T) *11.6
    energy2=(x*q/(mass*T) - x**2/(mass*T)-y**2/(mass*T) - q**2/(4*mass*T)+Cross*x*B/(mass*T)-Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T)+n/T-m/(2*T))*11.6
    numerator2 = np.exp(energy1)/ (np.exp(energy1)+ 1)
    numerator3 = 1 / (np.exp(energy2)+ 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2/numerator1
    elif energy1 < -700:
        return -numerator3/numerator1
    else:
        return (numerator2 - numerator3) / numerator1
    
def F(x, y,T,q,B, mass,Cross,m,n):
    numerator1 = (2*x**2/mass +2*y**2/mass+ q**2/(2*mass) - 2*n+Cross*2*x*B/mass+(Cross**2)*2*B**2/mass)
    energy1=(-x**2/(mass*T) - y**2/(mass*T)-q**2/(4*mass*T) - x*q/(mass*T)-Cross*x*B/(mass*T)-Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T)  +n/T-m/(2*T))*11.6
    energy2=(x*q/(mass*T) - x**2/(mass*T) -y**2/(mass*T)- q**2/(4*mass*T)-Cross*x*B/(mass*T)+Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T)+n/T+m/(2*T))*11.6
    numerator2 = np.exp(energy1)/ (np.exp(energy1)+ 1)
    numerator3 = 1 / (np.exp(energy2)+ 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2/numerator1
    elif energy1 < -700:
        return -numerator3/numerator1
    else:
        return (numerator2 - numerator3) / numerator1
def h(x, y,T,q,B,mass,Cross,m,n):
    numerator1 = (2*x**2/mass + 2*y**2/mass+q**2/(2*mass) - 2*n+m+Cross*2*q*B/mass+(Cross**2)*2*B**2/mass)
    energy1=(-x**2/(mass*T) -y**2/(mass*T)- q**2/(4*mass*T) - x*q/(mass*T) -Cross*x*B/(mass*T)-Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T) +n/T-m/(2*T))*11.6
    energy2=(x*q/(mass*T) - x**2/(mass*T) -y**2/(mass*T)- q**2/(4*mass*T)+Cross*x*B/(mass*T)-Cross*q*B/(2*mass*T)-(Cross**2)*B**2/(mass*T)+n/T-m/(2*T))*11.6
    numerator2 = np.exp(energy1)/ (np.exp(energy1)+ 1)
    numerator3 = 1 / (np.exp(energy2)+ 1)
    if abs(numerator1) < 1e-10:
        return 0
    elif energy1 > 700:
        return numerator2/numerator1
    elif energy1 < -700:
        return -numerator3/numerator1
    else:
        return (numerator2 - numerator3) / numerator1
matrix_v = np.full((4, 4), E)
matrix_v[0, 0] = A
matrix_v[3, 3] = A
matrix_v[0, 3] = J
matrix_v[3, 0] = J
matrix_V = -matrix_v

def matrix(T, q, B, mass,Cross, m, n):
    CC1 = integrate_2d(g, -2, 2, -2, 2, 0.01, 0.01, T,q,B,mass,Cross,m,n)
    CC2 = integrate_2d(f, -2, 2, -2, 2, 0.01, 0.01, T,q,B,mass,Cross,m,n)
    CC3 = integrate_2d(F, -2, 2, -2, 2, 0.01, 0.01, T,q,B,mass,Cross,m,n)
    CC4 = integrate_2d(h, -2, 2, -2, 2, 0.01, 0.01, T,q,B,mass,Cross,m,n)
    
    # 构造susceptibility矩阵matrix_X
    matrix_X = np.diag([CC1, CC2, CC3, CC4])
    product_matrix = np.dot(matrix_V, matrix_X)
    # 计算矩阵的本征值
    eigenvalues_product = np.linalg.eigvals(product_matrix)
    # 找到最大的本征值
    max_eigenvalue = np.max(eigenvalues_product)
    return max_eigenvalue

# 定义温度范围
T_values = np.linspace(1, 20, 20)

# 存储最大本征值
max_eigenvalues = []

# 计算每个温度下的最大本征值
for T in T_values:
    max_eigenvalue = matrix(T, q, B, mass,Cross, m, n)  # 假设q和B为0
    max_eigenvalues.append(max_eigenvalue)

# 绘制图表
plt.plot(T_values, max_eigenvalues, 'o')
plt.xlabel('T/K')
plt.ylabel('max_eigenvalue')
plt.title('f-T')
plt.grid(True)
plt.show()
end_time = time.time()
print(f"运行时间：{end_time - start_time}秒")