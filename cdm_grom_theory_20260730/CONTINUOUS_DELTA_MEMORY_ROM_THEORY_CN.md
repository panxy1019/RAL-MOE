# Continuous Delta-Memory Galerkin Reduced-Order Model

## 一套不使用神经网络的连续 Delta 记忆 Galerkin 降阶模型理论

## 摘要

本文提出 Continuous Delta-Memory Galerkin Reduced-Order Model
（CDM-GROM）。该方法面向 POD–Galerkin 截断后 resolved 模态动力学中
由未解析自由度引起的历史依赖。与固定三状态 delay window 不同，
CDM-GROM 使用固定维数内部状态保存衰减历史；与神经闭合不同，模型中
不存在 MoE、FNN、RNN、Transformer、attention、数据驱动 router 或外部
phase。模型参数由 Galerkin 算子、线性化 unresolved subsystem 或稳定
有理逼近构造。

方法从 channel-wise delta recurrence

\[
S_{n+1}
=(I-\beta_nk_nk_n^\top)D_nS_n+\beta_nk_nv_n^\top
\]

出发，在

\[
D_n=e^{-h_n\Gamma_n},\qquad
\beta_n=1-e^{-h_n\eta_n}
\]

的物理时间参数化下得到一阶连续极限

\[
\dot S=-\Gamma S+\eta k(v-S^\top k)^\top .
\]

将其与压力消元后的速度 POD–Galerkin 方程耦合，得到有限维增广
Markov 系统

\[
\dot a=F_G(a;\mu)+BS^\top q(a;\mu),\qquad
\dot S=-\Gamma(a,\mu)S+
\eta(a,\mu)k(a,\mu)[v(a,\mu)-S^\top k(a,\mu)]^\top .
\]

消去内部状态后，resolved 变量满足带历史积分的非马尔可夫方程。
在固定地址情况下，矩阵记忆可约化为辅助状态，并等价于有限指数
Volterra 核。在线性或线性化 resolved–unresolved Galerkin 系统中，
该核对应精确消去未解析变量后得到的
\(A_{ru}e^{A_{uu}\tau}A_{ur}\)。本文同时给出离散—连续一致性、
梯度流解释、局部适定性、memory 有界性、固定地址最小实现、能量稳定
互易子类以及核逼近误差传播的严格表述。

本文不声称连续模型与有限步长离散 recurrence 完全等价，也不把
memory 有界性解释为完整 ROM 的全局稳定性。线性 Galerkin 消元只在线性
或线性化系统中精确，能量稳定互易子类也不保证能够表达所有 Hopf
临界增长机制。

---

## 1. 方法定位

### 1.1 科学问题

设不可压缩流场经过 POD 投影后保留 \(r\) 个速度模态。若完整高维
Galerkin 状态写成 resolved 变量 \(a\) 与 unresolved 变量 \(c\)，则消去
\(c\) 通常会在 \(a\) 的方程中产生：

1. 依赖未解析初值的 initial-slip 项；
2. 对过去 resolved 轨迹的卷积记忆项；
3. 在线性化之外更一般的非线性历史算子。

固定历史窗口只保存若干离散快照，记忆长度随窗口大小固定，并依赖数据
采样间隔。CDM-GROM 改为引入连续物理时间中的内部状态，使记忆时间尺度
由衰减算子决定，而不是由 token 数决定。

### 1.2 模型边界

本理论版本只保留：

- POD–Galerkin resolved dynamics；
- 不可压缩流中的代数 Pressure–Poisson 关系；
- 由离散 channel-wise delta recurrence 导出的连续内部记忆；
- 由算子构造的固定矩阵。

本版本明确不包含：

- 任何神经网络或可训练 feature encoder；
- MoE、attention、router 和专家正则项；
- 三状态拼接、外部 phase 和未来元数据；
- 吸引子损失、normal-form head 和能量投影；
- 本轮尚未引入的旋转记忆块。

因此，CDM-GROM 应理解为一种“Galerkin 动力学 + 解析有限状态记忆闭合”，
而不是一种 attention 网络。

---

## 2. 问题表述与 Galerkin 降阶

### 2.1 参数化流动与 POD 坐标

令参数域 \(\mathcal P\subset\mathbb R^{d_\mu}\) 为紧集，\(\mu\in\mathcal P\)。
速度与压力分别采用 POD 展开

\[
u_r(x,t;\mu)=\bar u(x)+\sum_{i=1}^{r}a_i(t;\mu)\varphi_i(x),
\]

\[
p_{r_p}(x,t;\mu)=\bar p(x)+
\sum_{\ell=1}^{r_p}b_\ell(t;\mu)\psi_\ell(x).
\]

其中 \(a(t;\mu)\in\mathbb R^r\)，
\(b(t;\mu)\in\mathbb R^{r_p}\)。速度基在相应质量内积下正交归一。
半离散 Galerkin 系统写成

\[
\dot a=f_G(a,b;\mu).
\]

该向量场可以包含常数项、线性黏性项、二次对流项以及压力耦合项。
CDM-GROM 不改变这些已解析算子。

### 2.2 半显式压力处理

不可压缩 ROM 中的压力不引入独立动力学方程，而由离散
Pressure–Poisson 关系恢复：

\[
b=\mathcal P_p(a;\mu).
\]

因而压力消元后的速度场为

\[
F_G(a;\mu)
=f_G\!\left(a,\mathcal P_p(a;\mu);\mu\right).
\]

理论分析可直接使用 \(F_G\)。数值实现也可在一个接受的速度宏步后计算
\(b_{n+1}=\mathcal P_p(a_{n+1};\mu)\)。两种写法具有相同的代数职责：
pressure 没有独立时间导数。

### 2.3 截断导致的非马尔可夫闭合

若只观察 \(a\)，未解析模态的作用通常不能由当前 \(a(t)\) 唯一决定。
因此写成

\[
\dot a(t)=F_G(a(t);\mu)+r_{\rm mem}(t).
\]

CDM-GROM 的目标不是替换 \(F_G\)，而是用有限内部状态构造
\(r_{\rm mem}\)。

---

## 3. 从离散 Delta Recurrence 到连续时间记忆

### 3.1 离散 channel-wise recurrence

设矩阵状态 \(S_n\in\mathbb R^{d_k\times d_v}\)，写地址
\(k_n\in\mathbb R^{d_k}\)，目标 \(v_n\in\mathbb R^{d_v}\)，逐 key
通道遗忘矩阵 \(D_n\)。离散更新为

\[
S_{n+1}
=(I-\beta_nk_nk_n^\top)D_nS_n+\beta_nk_nv_n^\top.
\]

令

\[
\bar S_n=D_nS_n,
\]

则

\[
S_{n+1}
=\bar S_n+
\beta_nk_n(v_n-\bar S_n^\top k_n)^\top.
\]

第二种写法显示：先进行逐通道遗忘，再沿地址 \(k_n\) 写入当前
reconstruction error。

### 3.2 物理时间参数化

令 \(h_n=t_{n+1}-t_n>0\)，取

\[
D_n=e^{-h_n\Gamma_n},\qquad
\beta_n=1-e^{-h_n\eta_n},
\]

其中 \(\Gamma_n\succeq\gamma_{\min}I\)，\(\eta_n\ge0\)。在系数有界时，

\[
D_n=I-h_n\Gamma_n+O(h_n^2),
\qquad
\beta_n=h_n\eta_n+O(h_n^2).
\]

代入离散 recurrence 并保留一阶项：

\[
S_{n+1}=S_n+h_n
\left[-\Gamma_nS_n+
\eta_nk_n(v_n-S_n^\top k_n)^\top\right]+O(h_n^2).
\]

因此定义连续 delta-memory 方程

\[
\boxed{
\dot S=-\Gamma S+\eta k(v-S^\top k)^\top .
}
\]

这一推导给出单步 \(O(h_n^2)\) 的局部一致性。在标准 Lipschitz 与
稳定性条件下，固定有限时间区间上的全局误差是一阶 \(O(h)\)。
连续 ODE 与有限步长 recurrence 是两个不同模型；即使连续 ODE 使用
RK4 高阶积分，它与原离散 recurrence 的模型差异仍只有一阶趋于零。

### 3.3 CDM-GROM 增广系统

取固定解析特征

\[
\phi(a;\mu)=
[1,\ a^\top,\ (a\otimes a)^\top,\ \mu^\top]^\top
\in\mathbb R^{d_\phi}.
\]

由固定矩阵定义

\[
k(a;\mu)=\frac{K\phi(a;\mu)}
{\|K\phi(a;\mu)\|+\varepsilon},
\]

\[
v(a;\mu)=V\phi(a;\mu),\qquad
q(a;\mu)=\frac{Q\phi(a;\mu)}
{\|Q\phi(a;\mu)\|+\varepsilon}.
\]

\(\Gamma(a,\mu)\) 为正对角矩阵，\(\eta(a,\mu)\ge0\) 为固定解析函数或
算子派生函数。闭合读出为

\[
y=S^\top q(a;\mu),\qquad r_{\rm mem}=By.
\]

主模型为

\[
\boxed{
\begin{aligned}
\dot a
&=F_G(a;\mu)+BS^\top q(a;\mu),\\
\dot S
&=-\Gamma(a,\mu)S+
\eta(a,\mu)k(a;\mu)
[v(a,\mu)-S^\top k(a,\mu)]^\top .
\end{aligned}}
\]

在增广状态 \((a,S)\) 上，该系统是有限维 Markov ODE。消去 \(S\) 后，
\(a\) 的方程依赖过去轨迹，因此表现为非马尔可夫闭合。

### 3.4 正则化 reconstruction 梯度流

当 \(k,v,\Gamma,\eta>0\) 固定且 \(\Gamma\) 对称时，定义

\[
\mathcal J(S)
=\frac12\|v-S^\top k\|^2+
\frac{1}{2\eta}\operatorname{tr}(S^\top\Gamma S).
\]

其 Frobenius 梯度为

\[
\nabla_S\mathcal J
=-k(v-S^\top k)^\top+\eta^{-1}\Gamma S.
\]

所以

\[
\dot S=-\eta\nabla_S\mathcal J.
\]

这说明固定输入下的连续 delta-memory 是带 channel-wise Tikhonov
正则的在线 reconstruction 梯度流。

平衡点唯一：

\[
S_\star=
(\Gamma+\eta kk^\top)^{-1}\eta kv^\top,
\]

且误差满足

\[
S(t)-S_\star
=e^{-(\Gamma+\eta kk^\top)t}[S(0)-S_\star],
\]

从而指数收敛。

---

## 4. 有限状态与记忆核表示

### 4.1 固定地址下的最小状态

取固定单位地址

\[
k=q=e,\qquad \|e\|=1,\qquad
\Gamma=\gamma I,\qquad v(a)=Ca.
\]

定义有效读出

\[
y=S^\top e.
\]

则

\[
\dot y=-(\gamma+\eta)y+\eta Ca.
\]

另一方面，令 \(P_\perp=I-ee^\top\)，则

\[
\frac{d}{dt}(P_\perp S)=-\gamma P_\perp S.
\]

因此 \(e\) 正交补上的矩阵分量不影响读出，并按
\(e^{-\gamma t}\) 衰减。固定单地址时，完整矩阵 \(S\) 不是最小实现；
真正需要的状态只是 \(y\)。

多个正交固定地址可类似约化为多个辅助通道。该约化不适用于一般
state-dependent \(k(a)\) 和 \(q(a)\)，因为地址方向随轨迹变化。

### 4.2 辅助状态与 Volterra 核

考虑

\[
\dot a=F_G(a;\mu)+\sum_{j=1}^{m}B_jy_j,
\]

\[
\dot y_j=-\Lambda_jy_j+C_ja,
\qquad
\operatorname{Re}\sigma(\Lambda_j)>0.
\]

变参数公式给出

\[
y_j(t)=e^{-\Lambda_jt}y_j(0)
+\int_0^te^{-\Lambda_j(t-s)}C_ja(s)\,ds.
\]

代回 \(a\) 方程：

\[
\dot a(t)=F_G(a(t);\mu)+g_m(t)
+\int_0^tK_m(t-s)a(s)\,ds,
\]

其中

\[
g_m(t)=\sum_{j=1}^{m}B_je^{-\Lambda_jt}y_j(0)
\]

是 initial-slip 项，

\[
\boxed{
K_m(\tau)=
\sum_{j=1}^{m}B_je^{-\Lambda_j\tau}C_j
}
\]

是有限指数矩阵核。其拉普拉斯变换为

\[
\widehat K_m(s)
=\sum_{j=1}^{m}B_j(sI+\Lambda_j)^{-1}C_j.
\]

因此辅助状态实现与有限维有理 Volterra 核等价。

### 4.3 线性 resolved–unresolved Galerkin 消元

考虑线性或在参考状态附近线性化的分块系统

\[
\begin{bmatrix}\dot a\\\dot c\end{bmatrix}
=
\begin{bmatrix}
A_{rr}&A_{ru}\\
A_{ur}&A_{uu}
\end{bmatrix}
\begin{bmatrix}a\\c\end{bmatrix}.
\]

未解析状态满足

\[
c(t)=e^{A_{uu}t}c(0)
+\int_0^te^{A_{uu}(t-s)}A_{ur}a(s)\,ds.
\]

代回 resolved 方程：

\[
\dot a(t)
=A_{rr}a(t)
+A_{ru}e^{A_{uu}t}c(0)
+\int_0^t
A_{ru}e^{A_{uu}(t-s)}A_{ur}a(s)\,ds.
\]

因此精确线性核为

\[
\boxed{
K_{\rm exact}(\tau)
=A_{ru}e^{A_{uu}\tau}A_{ur}.
}
\]

若 \(A_{uu}\) Hurwitz，则该核指数衰减。令

\[
y=c,\qquad
B=A_{ru},\qquad C=A_{ur},\qquad
\Lambda=-A_{uu},
\]

CDM 辅助系统在线性情况下精确恢复该核。

该结论只对线性或线性化系统精确，不能直接扩展为完整非线性
Navier–Stokes 的精确 Mori–Zwanzig 公式。

### 4.4 不依赖神经网络的算子构造

#### 方案 A：精确 unresolved realization

从扩展阶数 POD–Galerkin 系统获得
\(A_{rr},A_{ru},A_{ur},A_{uu}\)，直接保留

\[
y=c,\quad B=A_{ru},\quad C=A_{ur},\quad
\Lambda=-A_{uu}.
\]

其优点是线性核精确；缺点是辅助维数等于全部未解析维数。

#### 方案 B：降阶 unresolved realization

对 unresolved subsystem 使用：

1. 谱截断；
2. 平衡截断；
3. rational Krylov；
4. 保稳定投影。

得到低维 \((\Lambda_j,B_j,C_j)\)。必须显式检查降阶后的
\(\operatorname{Re}\sigma(\Lambda_j)>0\)。

#### 方案 C：稳定 pole–residue 逼近

逼近 transfer function

\[
G_{\rm mem}(s)
=A_{ru}(sI-A_{uu})^{-1}A_{ur}
\]

为

\[
\sum_jB_j(sI+\Lambda_j)^{-1}C_j.
\]

所有极点必须位于连续系统的稳定半平面。若使用最小二乘或 vector
fitting，可称为 operator/data-assisted identification，但不能称为
neural learning。

对于当前方柱 Hopf 数据，现有产物只保存了 rank-11 POD 基。若采用方案
A 或 B，需要从原始快照重新构造更高阶 POD 空间和 resolved/unresolved
分块 Galerkin 算子；现有 rank-11 文件本身无法提供 \(A_{uu}\)。

---

## 5. 数学分析

### 5.1 局部适定性

在 A1–A3 下，增广向量场

\[
\mathcal F(a,S;\mu)=
\begin{bmatrix}
F_G(a;\mu)+BS^\top q(a;\mu)\\
-\Gamma(a,\mu)S+
\eta(a,\mu)k(a,\mu)[v(a,\mu)-S^\top k(a,\mu)]^\top
\end{bmatrix}
\]

对 \((a,S)\) 局部 Lipschitz。因而对任意有限初值存在唯一极大局部解。
该结论不自动给出全局存在：若 \(a\) 发生有限时间逃逸，memory 的单独
耗散不能阻止完整状态失稳。

### 5.2 Memory 有界性

令

\[
y=S^\top k.
\]

由 memory 方程可得精确能量恒等式

\[
\frac12\frac{d}{dt}\|S\|_F^2
=-\langle S,\Gamma S\rangle_F+
\eta y^\top(v-y).
\]

使用

\[
y^\top v-\|y\|^2
=-\left\|y-\frac12v\right\|^2+\frac14\|v\|^2
\]

以及 A5–A7：

\[
\frac12\frac{d}{dt}\|S\|_F^2
\le
-\gamma_{\min}\|S\|_F^2+
\frac{\eta_{\max}v_{\max}^2}{4}.
\]

因此

\[
\|S(t)\|_F^2
\le e^{-2\gamma_{\min}t}\|S(0)\|_F^2+
\frac{\eta_{\max}v_{\max}^2}{4\gamma_{\min}}
(1-e^{-2\gamma_{\min}t}).
\]

该估计在耦合解的整个存在区间上成立。若耦合解对所有 \(t\ge0\)
全局存在，则进一步有

\[
\limsup_{t\to\infty}\|S(t)\|_F^2
\le
\frac{\eta_{\max}v_{\max}^2}{4\gamma_{\min}}.
\]

另一种 Young 不等式给出 passivity/ISS 形式：

\[
\frac12\frac{d}{dt}\|S\|_F^2
\le
-\gamma_{\min}\|S\|_F^2
-\frac{\eta}{2}\|y\|^2
+\frac{\eta}{2}\|v\|^2.
\]

这表明 memory 对输入 \(v\) 是耗散的。它仍然不是完整
\((a,S)\) 系统的全局稳定性证明。

### 5.3 离散—连续一致性

若沿精确轨迹的系数足够光滑，Lemma 1 的展开余项在紧时间区间上一致，
则将精确 \(S(t_n)\) 代入离散 recurrence 后所得
\(S_{n+1}^{\rm rec}\) 满足
\[
\|S_{n+1}^{\rm rec}-S(t_{n+1})\|_F=O(h_n^2).
\]
若连续向量场在包含精确与数值轨迹的紧集上 Lipschitz，令
\(h=\max_nh_n\)，离散 Gronwall 不等式给出

\[
\max_{t_n\le T}\|S_n-S(t_n)\|_F\le C_Th.
\]

常数 \(C_T\) 依赖于时间区间、局部 Lipschitz 常数和一致二阶余项界，
但不依赖于 \(h\)。

### 5.4 能量稳定互易子类

考虑

\[
\dot a=F_G(a)-\sum_{j=1}^{m}C_j^\top y_j,
\]

\[
\dot y_j=\eta_jC_ja-\lambda_jy_j,
\qquad \eta_j,\lambda_j>0.
\]

定义增广能量

\[
\mathcal E(a,y)
=\frac12\|a\|^2+
\sum_{j=1}^{m}\frac{1}{2\eta_j}\|y_j\|^2.
\]

直接计算：

\[
\dot{\mathcal E}
=a^\top F_G(a)
-\sum_j a^\top C_j^\top y_j
+\sum_j\frac1{\eta_j}
y_j^\top(\eta_jC_ja-\lambda_jy_j).
\]

交叉项严格抵消，得到

\[
\boxed{
\dot{\mathcal E}
=a^\top F_G(a)
-\sum_{j=1}^{m}\frac{\lambda_j}{\eta_j}\|y_j\|^2.
}
\]

若 \(a^\top F_G(a)\le0\)，则在解的存在区间内 \(\mathcal E\) 非增。
若另外满足 A2，且更强地有

\[
a^\top F_G(a)\le c_0-c_1\|a\|^2,
\]

令

\[
\kappa=2\min\{c_1,\lambda_1,\ldots,\lambda_m\},
\]

则

\[
\dot{\mathcal E}\le c_0-\kappa\mathcal E,
\]

则能量有界排除有限时间状态逃逸，局部解可以延拓为全局解，并且

\[
\mathcal E(t)\le
e^{-\kappa t}\mathcal E(0)+
\frac{c_0}{\kappa}(1-e^{-\kappa t}).
\]

该子类产生的核为

\[
K_m(\tau)=
-\sum_j\eta_jC_j^\top e^{-\lambda_j\tau}C_j,
\]

主要是耗散型核，因此可能过度限制 Hopf onset 附近所需的增长机制。

### 5.5 核逼近误差传播

比较

\[
\dot a
=F_G(a)+g(t)+\int_0^tK(t-s)a(s)\,ds
\]

和

\[
\dot a_m
=F_G(a_m)+g_m(t)+
\int_0^tK_m(t-s)a_m(s)\,ds.
\]

假设：

- \(F_G\) 在共同解轨迹所在球内 Lipschitz，常数为 \(L\)；
- \(\|a_m(t)\|\le M\)；
- \(K,K_m\in L^1(0,T)\)；
- \(g-g_m\in L^1(0,T)\)。

定义

\[
\delta_K(T)=\int_0^T\|K(\tau)-K_m(\tau)\|\,d\tau,
\]

\[
\delta_g(T)=\int_0^T\|g(t)-g_m(t)\|\,dt,
\qquad
\kappa_T=\int_0^T\|K(\tau)\|\,d\tau.
\]

则 Volterra–Gronwall 估计给出

\[
\boxed{
\sup_{0\le t\le T}\|a(t)-a_m(t)\|
\le
\left[
\|a(0)-a_m(0)\|
+\delta_g(T)+MT\delta_K(T)
\right]
e^{(L+\kappa_T)T}.
}
\]

所以可取

\[
C_T=e^{(L+\kappa_T)T}\max\{1,MT\}.
\]

这明确说明误差常数如何依赖 \(T\)、Galerkin 局部 Lipschitz 常数、
精确核的 \(L^1\) 范数和近似解界。

---

## 6. 定理体系及依赖

本文正式英文稿包含：

1. **Lemma 1**：矩阵指数与 \(\beta\) 的一阶展开；
2. **Theorem 1**：离散 recurrence 的单步 \(O(h^2)\) 一致性；
3. **Corollary 1**：有限时间全局 \(O(h)\) 误差；
4. **Proposition 1**：正则 reconstruction 梯度流；
5. **Corollary 2**：固定输入下唯一平衡与指数收敛；
6. **Theorem 2**：增广系统局部存在唯一性；
7. **Theorem 3**：memory-state 显式有界性；
8. **Proposition 2**：memory 耗散/ISS 不等式；
9. **Proposition 3**：固定地址最小状态实现；
10. **Theorem 4**：辅助状态与 Volterra 核等价；
11. **Theorem 5**：线性 resolved–unresolved 精确消元；
12. **Corollary 3**：精确或降阶 kernel realization；
13. **Theorem 6**：能量稳定互易子类；
14. **Theorem 7**：kernel approximation error propagation。

完整依赖关系见 `THEOREM_DEPENDENCY_MAP.md`，假设定义见
`NOTATION_AND_ASSUMPTIONS.md`。

---

## 7. 理论 sanity checks

五个独立脚本分别检查：

1. 离散 recurrence 与连续向量场之间的二阶单步缺陷；
2. reconstruction objective 的有限差分梯度与解析梯度；
3. memory 能量恒等式、上界和 reciprocal coupling 交叉项抵消；
4. 固定 key 下矩阵方程与最小向量状态方程；
5. 辅助状态、卷积核和线性分块消元的一致性。

这些脚本只验证代数和维度，不构成流动数值实验，也不替代理论证明。

---

## 8. Scope and limitations

1. 正对角 \(\Gamma\) 主要生成衰减型时间尺度。
2. 纯实指数核不能高效表达强振荡记忆；Hopf 区域可能需要后续加入
   \(2\times2\) rotational blocks，但本轮不加入。
3. state-dependent \(k,q\) 产生非线性 fading-memory operator，此时
   不能再写成简单的线性 convolution kernel。
4. 固定地址时，矩阵 \(S\) 可能不是最小 realization。
5. memory boundedness 不等于 full-ROM stability。
6. 线性 resolved–unresolved 对应不能宣称为非线性精确性。
7. reciprocal energy-stable subclass 可能压制真实 Hopf 增长。
8. 若扩展 Galerkin 的 \(A_{uu}\) 不是 Hurwitz，直接把它作为稳定 memory
   realization 不满足 A8，必须先分离不稳定 resolved 方向或使用稳定投影。
9. 对一般非正规 \(\Lambda_j\)，特征值实部为正保证指数稳定，但简单
   Euclidean 能量单调性需要其对称部分正定，或使用 Lyapunov 权矩阵。
10. 本文没有声称首次在 ROM 中引入 memory；相关文献位置保留
    `[TODO citation: finite-memory/Mori–Zwanzig ROM literature]`。

---

## 9. 面向当前方柱 Hopf 项目的实施前提

当前 CenteredSquare Hopf 数据合同为：

- resolved 速度/压力 POD 阶数均为 11；
- 23 个完整 Re 工况用于 POD 与训练统计；
- 6 个完整 Re 工况用于验证；
- 5 个 held-out Re 在模型冻结前完全隔离。

纯算子 CDM-GROM 不需要使用已有神经训练标签，但构造
resolved–unresolved kernel 需要比 rank-11 更大的 POD 空间。建议后续：

1. 从虚拟机原始快照重新建立 \(r_{\rm full}=32,64,96\) 的 train-only
   POD 基；
2. 保留前 11 个速度模态为 resolved，其余为 unresolved；
3. 在同一网格排序和同一压力 gauge 下构建扩展 Galerkin/Pressure–
   Poisson 算子；
4. 在每个参考 Re 或参数插值节点上检查 \(A_{uu}\) 的谱；
5. 先完成线性 kernel 与 auxiliary realization 的算子级验证，再进行
   Hopf 轨迹积分。

由于此前方柱数据的原生快照时间间隔使直接连续 Galerkin RK4 主干出现
数值膨胀，后续不能默认原生宏步足够小。必须独立开展：

- Galerkin 主干时间步收敛；
- memory 子步与主状态联合积分；
- 原生步长、二分步长和四分步长一致性；
- 不含 memory 的扩展 Galerkin 基线稳定性检查。

---

## 10. 准确的结论

CDM-GROM 的理论核心不是把 KDA 名称移植到 ROM，而是建立以下链条：

\[
\text{channel-wise delta recurrence}
\longrightarrow
\text{continuous internal state}
\longrightarrow
\text{finite exponential Volterra kernel}
\longrightarrow
\text{resolved–unresolved Galerkin realization}.
\]

其优势是：

- 历史以物理时间衰减，而非固定 token 数截断；
- 内部状态维数固定；
- 参数可以完全由 Galerkin 算子构造；
- memory 子系统具有显式有界性；
- 特定 reciprocal 结构可获得增广能量估计；
- kernel 逼近误差可以传播到 resolved 解误差。

其理论边界同样明确：一般非线性 CDM-GROM 只保证局部适定性；
memory 有界不保证完整 ROM 全局稳定；线性 kernel 对应只在线性或
线性化系统中精确；纯耗散 reciprocal 子类不必然再现 Hopf 临界增长。
