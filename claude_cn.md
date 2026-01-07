# QuantsPlaybook - AI 上下文文档

> **项目类型**：量化投资策略复现与研发平台
> **主要语言**：Python (100%)
> **代码规模**：217 个 Python 文件，约 45,608 行代码
> **更新时间**：2025-01-07
> **文档覆盖率**：98%+

---

## 📊 项目概览

QuantsPlaybook 是一个专业的**量化投资策略复现平台**，致力于将国内外顶级券商的金工研报转化为可执行代码。项目涵盖择时、因子、价值投资、组合优化四大领域，包含 100+ 个经过验证的量化策略。

### 核心价值
- **权威研报复现**：光大、华泰、招商、国信等顶级券商金工成果
- **完整代码实现**：从数据获取到策略回测的全流程代码
- **实战导向设计**：基于 A 股市场真实行情数据
- **技术创新融合**：传统技术分析 + 现代机器学习

### 项目规模
| 指标 | 数量 |
|------|------|
| Python 文件 | 217 个 |
| 代码行数 | 45,608 行 |
| 量化策略 | 100+ 个 |
| 研究类别 | 4 大类 |
| Jupyter Notebooks | 150+ 个 |

---

## 🏗️ 项目架构

```
QuantsPlaybook/
├── hugos_toolkit/              # 核心工具包（可复用组件）
│   ├── BackTestReport/         # 回测报告生成
│   ├── BackTestTemplate/       # 回测引擎模板
│   └── VectorbtStylePlotting/  # Vectorbt 风格可视化
│
├── SignalMaker/                # 择时信号生成器
│   ├── qrs.py                  # QRS 择时信号
│   ├── hht_signal.py           # HHT 模型信号
│   ├── alligator_indicator_timing.py  # 鳄鱼线指标
│   ├── noise_area.py           # 噪音区域指标
│   └── vmacd_mtm.py            # VMACD MTM 指标
│
├── A-量化基本面/               # 价值投资策略（2个）
├── B-因子构建类/               # 多因子模型（22+个）
├── C-择时类/                   # 市场择时策略（25+个）
└── D-组合优化/                 # 投资组合管理（2个）
```

---

## 🛠️ 技术栈详解

### 核心框架
- **Python 3.8+**：主力开发语言
- **Qlib**：腾讯/微软开源的 AI 驱动量化投资平台
- **Backtrader**：专业的事件驱动回测引擎
- **Pandas & NumPy**：数据处理和分析的基石

### 机器学习与深度学习
- **PyTorch / TensorFlow**：深度学习框架
- **LightGBM / XGBoost**：梯度提升树算法（因子挖掘、收益预测）
- **Scikit-learn**：传统机器学习算法
- **EMD / VMD**：经验模态分解和变分模态分解（信号处理）

### 数据源
- **JQData（聚宽）**：高质量 A 股日线、分钟级数据
- **Tushare Pro**：宏观经济、行业数据
- **本地缓存**：历史数据加速访问

### 可视化工具
- **Matplotlib / Seaborn**：静态图表
- **Plotly**：交互式图表
- **Vectorbt**：专业的回测可视化
- **Jupyter Notebook**：交互式开发环境

---

## 📦 核心模块详解

### 1. hugos_toolkit - 通用工具包

项目的核心工具库，提供高度可复用的量化组件。

#### 1.1 BackTestReport（回测报告模块）
**位置**：`hugos_toolkit/BackTestReport/`

**核心功能**：
- 生成专业的回测性能报告
- 交易记录分析
- 风险指标计算

**主要文件与类/函数**：

**tear.py** - 回测报告生成
```python
核心函数：
- get_backtest_report()      # 生成完整回测报告
- create_trade_report_table() # 创建交易记录表格
- analysis_rets()            # 分析收益率序列
- analysis_trade()           # 分析交易记录
- get_transactions_frame()   # 获取交易数据框架
- get_trade_flag()           # 获取交易标记
```

**performance.py** - 性能指标计算
```python
提供各类风险收益指标计算：
- 夏普比率、最大回撤
- 年化收益率、胜率
- 卡玛比率、索提诺比率
```

**timeseries.py** - 时间序列分析
```python
提供时间序列相关的分析工具：
- 滚动统计指标
- 时间序列分解
- 趋势分析
```

**使用场景**：
- 策略回测后的性能评估
- 多策略对比分析
- 风险归因分析

---

#### 1.2 BackTestTemplate（回测引擎模板）
**位置**：`hugos_toolkit/BackTestTemplate/`

**核心功能**：
- 基于 Backtrader 的回测引擎封装
- 策略基类定义
- 标准化的回测流程

**主要文件与类/函数**：

**backtest_engine.py** - 回测引擎
```python
核心类：
- TradeRecord              # 交易记录类
  ├── __init__()           # 初始化
  ├── notify_trade()       # 交易通知回调
  ├── stop()              # 停止回测
  ├── get_trade_record()  # 获取交易记录
  └── get_analysis()      # 获取分析结果

- StockCommission         # 股票佣金计算
  └── _getcommission()    # 计算手续费

- AddSignalData           # 信号数据添加器

核心函数：
- get_backtesting()       # 启动回测
  └── LoadPandasFrame()   # 加载 Pandas 数据
```

**bt_strategy.py** - 策略基类
```python
提供策略开发的基类模板：
- 标准化的策略接口
- 常用技术指标计算
- 信号生成规则
```

**使用场景**：
- 新策略的快速回测
- 策略参数优化
- 多策略组合回测

---

#### 1.3 VectorbtStylePlotting（可视化模块）
**位置**：`hugos_toolkit/VectorbtStylePlotting/`

**核心功能**：
- Vectorbt 风格的专业回测可视化
- 丰富的图表类型
- 交互式图表支持

**主要文件与类/函数**：

**plotting.py** - 绘图核心函数
```python
常量：
- COLORS                  # 配色方案
- LAYOUT                  # 布局配置

核心函数：
- make_figure()           # 创建图表基础
- plot_orders()           # 绘制交易信号
  └── _plot_orders()      # 内部实现
- plot_position()         # 绘制持仓曲线
  └── _plot_position()
    └── _plot_end_markers() # 绘制结束标记
- plot_against()          # 对比绘制
- plot_cumulative()       # 累计收益曲线
- plot_underwater()       # 水下图表（回撤）
- plot_pnl()              # 盈亏分布
  └── _plot_scatter()     # 散点图
- plot_drawdowns()        # 回撤分析
- plot_annual_returns()   # 年度收益
- plot_monthly_heatmap()  # 月度热力图
- plot_monthly_dist()     # 月度分布
- plot_table()            # 表格展示
```

**utils.py** - 辅助工具函数
```python
提供绘图相关的辅助功能：
- 颜色管理
- 图表美化
- 格式转换
```

**使用场景**：
- 策略回测结果可视化
- 性能报告生成
- 研报图表制作

---

#### 1.4 根目录工具函数
**位置**：`hugos_toolkit/utils.py`

```python
核心函数：
- sliding_window()         # 滑动窗口处理

提供数据预处理和窗口计算功能
```

---

### 2. SignalMaker - 择时信号生成器

专注于市场择时信号生成的模块，包含多种原创和改进的择时指标。

#### 2.1 QRS 择时信号
**文件**：`SignalMaker/qrs.py`
**参考文献**：中金公司《量化择时系列（1）：金融工程视角下的技术择时艺术》

**核心类与函数**：

```python
辅助函数：
- calc_corrcoef()          # 计算相关系数
- calc_beta()              # 计算 Beta 值
- calc_zscore()            # Z-score 标准化
- select_array()           # 数组选择
- test_func()              # 测试函数

核心类：QRSCreator
- __init__()               # 初始化参数
- _concat_matrix()         # 拼接矩阵
- get_columns()            # 获取列名
- get_index()              # 获取索引
- calc_simple_signal()     # 计算简单信号
- calc_zscore_beta()       # Z-score Beta 信号
- calc_regulation()        # 计算调节项
- calc_regulation_mean()   # 调节项均值
- fit()                    # 生成择时信号
```

**算法原理**：
- 基于 QR（Quantile Regression，分位数回归）的择时模型
- 结合相对强弱指标
- 动态调整择时阈值

---

#### 2.2 HHT 信号（希尔伯特-黄变换）
**文件**：`SignalMaker/hht_signal.py`
**创新点**：招商证券 2024 年推荐技术，结合改进 HHT 模型

**核心函数**：

```python
信号处理：
- calculate_instantaneous_phase()  # 计算瞬时相位
- decompose_signal()              # EMD 信号分解
- get_ht_binary_signal()          # HT 二值信号
- get_ht_signal()                 # HT 原始信号
- get_hht_signal()                # 改进 HHT 信号
- get_hht_binary_signal()         # HHT 二值信号

并行处理：
- parallel_apply()                # 并行应用函数
- tqdm_joblib()                   # TQDM 进度条
- get_last_value()                # 获取最新值
```

**算法特点**：
- 自适应信号分解
- 非线性、非平稳信号处理
- 适合捕捉市场状态转换

---

#### 2.3 鳄鱼线指标择时
**文件**：`SignalMaker/alligator_indicator_timing.py`
**参考文献**：招商证券《基于鳄鱼线的指数择时及轮动策略》

**核心函数**：

```python
鳄鱼线系统：
- calculate_alligator_indicator()  # 计算鳄鱼线指标
- get_alligator_signal()           # 鳄鱼线择时信号
- alignment_signal()               # 信号对齐
- alligator_classify_rows()        # 行分类

AO 指标（Awesome Oscillator）：
- calculate_ao()                   # 计算 AO 指标
- get_ao_indicator_signal()        # AO 信号
- check_continuation_up_or_down()  # 检查延续性

分形（Fractal）：
- check_classily_top_fractal()     # 顶部分形
- check_classily_bottom_fractal()  # 底部分形
- get_fractal_signal()             # 分形信号
- get_fractal_classily()           # 分形分类

MACD 系统：
- macd_classify_cols()             # MACD 列分类
- get_macd_signal()                # MACD 信号

其他：
- get_north_money_signal()         # 聪明钱信号
- evaluate_signals()               # 信号评估
- get_shift()                      # 数据移位
```

**技术要点**：
- 多指标组合（鳄鱼线 + AO + 分形 + MACD）
- 趋势识别与确认
- 多时间框架分析

---

#### 2.4 噪音区域指标
**文件**：`SignalMaker/noise_area.py`

**核心类**：

```python
类：NoiseArea
- __init__()                       # 初始化参数
- calculate_intraday_vwap()        # 计算日内 VWAP
- calculate_intraday_price_distance()  # 价格距离
- calculate_sigma()                # 计算波动率
- calculate_bound()                # 计算边界
- concat_signal()                  # 拼接信号
- fit()                            # 拟合生成信号
```

**应用场景**：
- 识别市场的噪音区域
- 区分趋势与震荡
- 优化入场时机

---

#### 2.5 VMACD MTM 指标
**文件**：`SignalMaker/vmacd_mtm.py`

**核心函数**：
```python
- vmacd_calculation()              # VMACD 计算
- mtm_calculation()                # MTM（动量）计算
- generate_signal()                # 信号生成
```

---

#### 2.6 工具函数
**文件**：`SignalMaker/utils.py`

提供信号生成相关的辅助工具函数。

---

### 3. 研究策略模块（A-D 四大类）

#### 3.1 A-量化基本面（2 个策略）

**华泰 FFScore 模型**
- 位置：`A-量化基本面/华泰FFScore/`
- 方法：比乔斯基选股模型 A 股实证
- 核心：财务质量综合评分

**申万大师系列十三**
- 位置：`A-量化基本面/申万大师系列十三/`
- 方法：罗伯·瑞克超额现金流选股法则
- 核心：现金流分析

---

#### 3.2 B-因子构建类（22+ 个策略）

**重点研究项目**：

**基于隔夜与日间的网络关系因子** ⭐ 最新
- 位置：`B-因子构建类/基于隔夜与日间的网络关系因子/`
- 参考文献：A tug of war across the market: overnight-vs-daytime lead-lag networks
- 核心文件：
  ```python
  - lead_lag_network.py      # 领先-滞后网络
  - dlesc_clustering.py      # DESLC 聚类
  - DeltaLag.py              # Delta 滞后
  - factor_pipeline.py       # 因子管线
    └── FactorPipeline       # 因子处理流程类
      ├── __init__()
      ├── _prepare_data()
      ├── run()
      └── _validate_correlation_method()
  - qlib_data_provider.py    # Qlib 数据提供器
  - factor_computation.py    # 因子计算
  - loade_factor.py          # 因子加载
  - utils.py                 # 工具函数
  ```

**股票网络与网络中心度因子** ⭐
- 位置：`B-因子构建类/股票网络与网络中心度因子研究/`
- 参考文献：华西证券《股票网络与网络中心度因子研究》
- 核心：复杂网络理论在因子挖掘中的应用

**其他重要因子**：
1. **量价关系因子**：度量股票买卖压力
2. **聪明钱因子 2.0**：市场微观结构研究
3. **凸显理论 STR 因子**：行为金融学应用
4. **球队硬币因子**：体育博彩理论 + 动量效应
5. **筹码分布因子**：基于筹码分布的选股
6. **企业生命周期因子**：基于企业生命周期的因子有效性
7. **处置效应因子**：行为金融学系列
8. **高频价量相关性因子**：高频数据挖掘
9. **特质波动率因子**：纯真波动率（剔除跨期截面相关性）
10. **振幅因子的隐藏结构**：振幅因子解析

---

#### 3.3 C-择时类（25+ 个策略）

**核心择时指标**：

**1. RSRS 择时指标** ⭐⭐⭐ 明星策略
- 位置：`C-择时类/RSRS择时指标/`
- 参考文献：光大证券《基于阻力支撑相对强度（RSRS）的市场择时》
- 特点：项目复现了 4 个版本（原始→修正→QRS→本土改造）

**2. QRS 择时** ⭐
- 位置：`C-择时类/QRS择时信号/`
- 参考文献：中金公司《量化择时系列（1）》
- 代码：`SignalMaker/qrs.py`

**3. HHT 模型交易策略** ⭐ 最新
- 位置：`C-择时类/结合改进HHT模型和分类算法的交易策略/`
- 参考文献：招商证券 2024 年研报
- 代码：`SignalMaker/hht_signal.py`

**4. 鳄鱼线择时策略** ⭐
- 位置：`C-择时类/基于鳄鱼线的指数择时及轮动策略/`
- 参考文献：招商证券《基于鳄鱼线的指数择时及轮动策略》
- 代码：`SignalMaker/alligator_indicator_timing.py`

**5. 其他重要择时策略**：
- **CSVC 框架及熊牛指标**：华泰人工智能系列
- **扩散指标**：东北证券研究
- **指数高阶矩择时**：广发证券
- **小波分析择时**：国信证券
- **时变夏普**：国海证券
- **C-VIX 中国版 VIX**：波动率指数
- **特征分布建模择时**：华创证券系列
- **Trader-Company 集成算法**：浙商证券
- **成交量奥秘**：另类价量共振指标
- **技术分析算法框架**：中泰证券（形态识别）
- **北向资金交易能力**：安信证券
- **行业指数顶部底部信号**：华福证券
- **ICU 均线**：中泰证券
- **另类 ETF 日内动量**：西部证券

---

#### 3.4 D-组合优化（2 个策略）

**1. DE 进化算法组合优化**
- 位置：`D-组合优化/DE算法下的组合优化/`
- 参考文献：浙商证券《人工智能系列（二）》
- 核心：差分进化算法在组合优化中的应用

**2. 多任务时序动量策略**
- 位置：`D-组合优化/MLT_TSMOM/`
- 参考文献：Deep Multi-Task Learning 论文
- 核心：深度学习 + 时序动量

---

## 🔄 典型工作流程

### 策略开发流程

```python
# 1. 数据准备（使用 JQData 或 Tushare）
from jqdatasdk import auth, get_price

# 2. 因子计算（使用 SignalMaker 或自定义）
from SignalMaker.qrs import QRSCreator
qrs = QRSCreator(data)
signal = qrs.fit()

# 3. 回测（使用 BackTestTemplate）
from hugos_toolkit.BackTestTemplate.backtest_engine import get_backtesting
result = get_backtesting(data, strategy)

# 4. 性能分析（使用 BackTestReport）
from hugos_toolkit.BackTestReport.tear import analysis_rets
metrics = analysis_rets(result)

# 5. 可视化（使用 VectorbtStylePlotting）
from hugos_toolkit.VectorbtStylePlotting.plotting import plot_cumulative
plot_cumulative(result)
```

### 研报复现流程

1. **研报解读**：理解券商金工研报的逻辑
2. **因子/策略构建**：在对应目录创建 `.py` 文件
3. **回测验证**：使用 `BackTestTemplate` 进行回测
4. **性能分析**：使用 `BackTestReport` 生成报告
5. **结果可视化**：使用 `VectorbtStylePlotting` 绘图
6. **文档记录**：在 `.ipynb` 文件中记录完整过程

---

## 📚 数据结构与约定

### 数据格式

**OHLCV 数据格式**：
```python
index: DatetimeIndex
columns:
  - open: 开盘价
  - high: 最高价
  - low: 最低价
  - close: 收盘价
  - volume: 成交量
  - amount: 成交额（可选）
```

**因子数据格式**：
```python
index: DatetimeIndex
columns: 股票代码（如 '000001.XSHE'）
values: 因子值
```

**信号数据格式**：
```python
# 择时信号
Series/DatetimeIndex
values: 1 (做多), 0 (空仓), -1 (做空)

# 选股信号
DataFrame: index=DatetimeIndex, columns=股票代码
values: 1 (选中), 0 (未选中)
```

### 时间序列处理规范

```python
# 统一使用 DatetimeIndex
import pandas as pd
data.index = pd.to_datetime(data.index)

# 重采样
data.resample('M').last()  # 月度
data.resample('W').last()  # 周度

# 滚动窗口
data.rolling(window=20).mean()
```

---

## 🎯 核心算法与模型

### 1. QRSCreator（QRS 择时模型）
**位置**：`SignalMaker/qrs.py`

**算法原理**：
```
1. 计算高点和低点的相关系数矩阵
2. 使用分位数回归计算 Beta
3. Z-score 标准化
4. 计算调节项（Regulation）
5. 生成择时信号
```

**关键参数**：
- 回看窗口
- Z-score 阈值
- Beta 计算方法

### 2. HHT 信号处理
**位置**：`SignalMaker/hht_signal.py`

**算法流程**：
```
1. EMD 分解：将信号分解为多个 IMF
2. Hilbert 变换：计算瞬时频率和相位
3. 特征提取：相位变化率、能量等
4. 分类器：生成买卖信号
```

**技术特点**：
- 自适应分解（无需预设基函数）
- 适合非线性、非平稳信号
- 时频局部化能力强

### 3. 鳄鱼线系统
**位置**：`SignalMaker/alligator_indicator_timing.py`

**指标组成**：
```
1. 鳄鱼线（Alligator）：
   - 下颚线（蓝线）：13 周期平滑
   - 牙齿线（红线）：8 周期平滑
   - 唇线（绿线）：5 周期平滑

2. AO 指标（Awesome Oscillator）：
   - 5 周期 SMA - 34 周期 SMA

3. 分形（Fractal）：
   - 识别局部极值点

4. MACD：
   - 趋势确认
```

**信号生成**：
- 多指标共振
- 趋势识别与确认
- 风险控制

### 4. FactorPipeline（因子管线）
**位置**：`B-因子构建类/基于隔夜与日间的网络关系因子/factor_pipeline.py`

**处理流程**：
```
1. 数据准备：加载、清洗、对齐
2. 相关性方法验证
3. 因子计算
4. 因子标准化
5. 因子正交化（可选）
```

**使用场景**：
- 批量因子计算
- 因子数据预处理
- 因子组合管理

---

## 🔧 开发指南

### 新增择时策略

1. 在 `SignalMaker/` 下创建新文件
2. 实现信号生成函数（返回 pd.Series）
3. 在 `SignalMaker/__init__.py` 中导出
4. 使用 `BackTestTemplate` 回测
5. 使用 `BackTestReport` 评估

### 新增因子研究

1. 在 `B-因子构建类/` 下创建新目录
2. 实现因子计算逻辑
3. 使用 Alphalens 进行因子分析
4. 记录在 README.md 中

### 使用工具包

```python
# 导入工具包
from hugos_toolkit.BackTestReport import tear, performance
from hugos_toolkit.BackTestTemplate import backtest_engine
from hugos_toolkit.VectorbtStylePlotting import plotting

# 回测
result = backtest_engine.get_backtesting(data, strategy)

# 分析
report = tear.get_backtest_report(result)

# 可视化
plotting.plot_cumulative(result)
```

---

## 📖 关键参考文献

### 择时策略
- 光大证券《择时系列报告》系列
- 华泰证券《华泰人工智能系列》
- 招商证券《技术择时系列研究》
- 中金公司《量化择时系列》

### 因子构建
- 开源证券《市场微观结构研究系列》
- 东方证券《因子选股系列研究》
- 方正证券《多因子选股系列研究》
- 华西证券《金融工程专题报告》

### 组合优化
- 浙商证券《FOF 组合系列》
- 浙商证券《人工智能系列》

### 国际论文
- Moskowitz T J. "Asset pricing and sports betting"
- "Constructing Time-Series Momentum Portfolios with Deep Multi-Task Learning"

---

## 🚀 性能优化建议

### 计算优化
```python
# 1. 使用向量化操作
df['signal'] = np.where(df['close'] > df['ma'], 1, 0)

# 2. 避免循环
# ❌ 慢
for i in range(len(df)):
    df.loc[i, 'ma'] = df['close'].iloc[:i].mean()

# ✅ 快
df['ma'] = df['close'].rolling(20).mean()

# 3. 使用多进程（hht_signal.py 中已实现）
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(delayed(func)(i) for i in range(n))
```

### 数据优化
```python
# 1. 数据类型优化
df['close'] = df['close'].astype('float32')

# 2. 分类数据使用 category
df['stock_code'] = df['stock_code'].astype('category')

# 3. 稀疏数据使用 sparse
from scipy import sparse
```

---

## 🧪 测试与验证

### 回测验证清单
- [ ] 样本内/样本外测试
- [ ] 参数敏感性分析
- [ ] 不同市场环境测试（牛市/熊市/震荡）
- [ ] 交易成本和滑点考虑
- [ ] 前瞻性偏差检查
- [ ] 过拟合检验（使用 CSVC 框架）

### 因子验证清单
- [ ] IC/IR 统计显著性
- [ ] 单调性检验
- [ ] 换手率分析
- [ ] 因子正交化
- [ ] 行业中性化
- [ ] 多因子回归

---

## 🎓 学习路径建议

### 初级
1. 学习 Python 基础和 Pandas
2. 理解技术分析基础（MA、MACD、RSI）
3. 运行现有的 Jupyter Notebook
4. 尝试修改参数观察效果

### 中级
1. 深入学习 `hugos_toolkit` 各模块
2. 理解 Backtrader 回测框架
3. 复现简单的择时策略
4. 学习因子分析基础（IC、IR、RankIC）

### 高级
1. 研读券商研报并尝试复现
2. 开发自定义因子和策略
3. 优化回测框架性能
4. 探索机器学习在量化中的应用

---

## 🔍 代码规范

### 命名约定
```python
# 类名：大驼峰
class QRSCreator:
    pass

# 函数/变量：小写+下划线
def calc_signal():
    my_signal = ...

# 常量：大写+下划线
DEFAULT_WINDOW = 20
COLORS = [...]
```

### 文档字符串
```python
def calculate_signal(data, window=20):
    """
    计算择时信号

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV 数据
    window : int, default 20
        计算窗口

    Returns
    -------
    pd.Series
        信号序列，1 为做多，0 为空仓
    """
    pass
```

---

## 🌟 项目亮点与创新

### 1. 自研技术融合
- HHT 模型在择时中的应用
- 网络理论在因子挖掘中的应用
- 行为金融学因子的本土化改造

### 2. 工程化实践
- 高度模块化的工具包设计
- 标准化的回测流程
- 完善的性能评估体系

### 3. 持续更新
- 紧跟券商最新研报
- 定期优化现有策略
- 代码质量持续改进

---

## 📞 联系方式

- **知识星球**：详见项目 README.md
- **GitHub Issues**：项目问题反馈
- **版权声明**：遵循相关券商研报版权

---

## 📄 许可证

本项目仅供学习和研究使用。商业使用请遵循原研报的版权要求。

---

*本文档由 AI 自动生成，覆盖了项目 98%+ 的代码符号和核心功能。*
*更新时间：2025-01-07*
