# Phase 1：Core 與 Screening 文件

範圍：Plackett–Burman 設計、主效應分析、t-limit / Bonferroni limit、逐步篩選、Pareto / Half-Normal / Absolute-effects 圖。
（Phase 1。Fractional Factorial 見 §4；RSM Design 以後屬 Phase 2，**尚未實作**。）

## 1. API

```python
import adore_doe as ad

factors = [ad.Factor("A", 0.1, 0.3, unit="g"), ad.Factor("B", 10, 20), ad.Factor("C", 1, 5)]
design  = ad.screening.plackett_burman(factors, runs=8, center_points=1, randomize=True, seed=1)
design.to_csv("design.csv")                       # 依 RunOrder 輸出；首行為可還原的 metadata
design.add_response("Yield", values, order="run") # order="std"(預設) 或 "run"（依實驗執行順序填）
effects   = ad.screening.main_effects(design, response="Yield", alpha=0.05)
screening = ad.screening.screen(effects, alpha=0.05, criterion="t_limit")
screening.plot_pareto(); screening.plot_half_normal(); screening.plot_abs_effects()
screening.selected_factors                        # -> RSM 階段
screening.export_selected("selected.json")        # 或 .csv
```

| 物件 | 重點 |
|---|---|
| `Response(name, weight, r2_threshold, direction, target, unit)` | 後續階段的聚合 / 停止 / 最佳化設定；本階段只驗證與保存 |
| `Experiment` | 單一 run 的唯讀檢視（`dataset.experiment(run_id)`） |
| `Factor(name, low, high, unit, factor_type, metadata)` | `to_coded` / `from_coded`；numeric：low→−1、high→+1、中心→0；categorical 只有 ±1，無中心點 |
| `ExperimentDataset` | `coded` / `actual_df()` / `to_dataframe(order="std"\|"run")`；多 response；`std_order`（設計列編號，不變）與 `run_order`（執行順序）分離；`validate()`；`to_csv` / `from_csv`；`randomize(seed)` |
| `plackett_burman(factors, runs=None, center_points=0, randomize=False, seed=None)` | 非法 `runs` 一律報錯，不會被改成別的值；`runs=None` 取最小合法值；未使用的欄位保留為 dummy columns |
| `main_effects(design, response, alpha, curvature=True)` → `EffectsResult` | `.table`、`.df_error`、`.t_limit`、`.bonferroni_limit`、`.lenth`、`.curvature`、`.warnings`、`.plot_*` |
| `screening(effects, alpha, method, criterion, direction, start, add, max_steps, min_df)` → `ScreeningResult` | `.selected_factors`、`.candidates`、`.bonferroni_selected`、`.iterations`（稽核紀錄）、`.history()`、`.final_table`、`.stop_reason`、`.notes` |
| `analysis.stepwise_selection(X, y, names, ...)` | 與圖表、設計容器完全獨立的篩選 API（純陣列） |
| `analysis.limits` | `t_limit`、`bonferroni_limit`、`lenth`、`classify` |
| `analysis.halfnormal` | `half_normal_quantiles`、`half_normal_points`、`reference_slope` |

所有隨機性只在 `randomize`，由 `seed` 控制；未給 seed 時會產生一個並記錄於 `design.info["seed"]`，可重現。

## 2. 統計定義

編碼：`x = (X − (high+low)/2) / ((high−low)/2)`。模型 `y = β0 + Σ βj xj [+ βc·1(center)] + ε`，OLS 估計。

| 量 | 公式 |
|---|---|
| Effect | `Eⱼ = 2βⱼ`。平衡正交設計下 = `mean(y\|xⱼ=+1) − mean(y\|xⱼ=−1)`（`Mean_Diff` 欄；兩者不同時警告並採 OLS 值） |
| Coefficient | `βⱼ = Eⱼ/2`（每個 coded 單位）。**與 Effect 差 2 倍，兩欄都輸出以免混淆** |
| 誤差變異 | `σ² = RSS/df`，`df = n − rank(Z)` |
| SE | `SE(Eⱼ) = 2·SE(βⱼ)`，`SE(βⱼ)² = σ²[(Z′Z)⁻¹]ⱼⱼ` |
| t / p | `t = Eⱼ/SE(Eⱼ) = βⱼ/SE(βⱼ)`；`p = 2·P(T_df > |t|)` |
| CI | `Eⱼ ± t_{1−α/2, df}·SE(Eⱼ)` |
| Standardized effect | `Effect / SE(Effect)`（= t），與 t-limit / Bonferroni limit **同尺度**。Design-Expert 半常態圖 X 軸的 "standardized effect" 內部縮放不同，**不宣稱等價** |

**誤差自由度的來源**：(1) center points 的純誤差（nc − 1，另含一個 curvature 項）；(2) PB 未指派的 dummy columns（假設無交互作用）；(3) 多出的 runs。`df < 1` 時**不產生** SE / t / p / CI（輸出 NaN 並警告）。

**Curvature**：有 center points 時加入 `βc·1(center)`，估計值在正交設計下為 `mean(center) − mean(factorial)`，並給出 t / p；曲率不會灌水 σ。`curvature=False` 不建議（σ 被曲率膨脹）。

**不可估計**：設計矩陣不滿秩（如缺值使欄位混淆、重複欄位）→ `RankDeficientError`，訊息列出不可估計的參數，不靜默刪欄。缺失 response 會被丟棄並警告。

### 2.1 limits（`screening/limits.py`）

三件事是分開的：

1. **單一因子檢定**：`t_limit = t_{1−α/2, df}`（雙尾）。每個因子單獨 type-I error = α。
2. **多重比較校正**：`bonferroni_limit = t_{1−α/(2m), df}`，m = 同時檢定的因子數。這是對 m 個雙尾 t 檢定各用 α/m 的標準 Bonferroni（union bound，FWER ≤ α）。
3. **逐步篩選**（§2.2）。

與 PDF 第 1 頁對照（α = 0.05）：df = 10 → t-limit = **2.22814**；df = 10、m = 19 → Bonferroni = **3.97276**。測試 `test_pdf_page1_configuration_reproduces_reference_limits` 以 PB-24（19 因子）+ 7 個 center points（df = 31−1−19−1 = 10）重現這組設定。

> 驗證範圍聲明：m = 19 是由 PDF 圖上的數值與 Rank 軸反推，並以 SciPy 確認一致；**沒有 Design-Expert 的原始輸出檔，因此未做 Design-Expert 數值回歸**，只驗證 limit 公式與圖上兩個數字吻合。

### 2.2 逐步篩選（`screening/selection.py`）

目前模型 M：OLS（含截距、必要時含 curvature）→ `σ_M`、`df_M`。
1. 每個因子算 *partial t*（M 內的因子：扣除自身；M 外：加入 M），一律使用 `σ_M`，因此全部 t 在同一尺度、可共用一條 limit。
2. 依 `criterion` 取 t-limit 或 Bonferroni limit（皆用 `df_M`；m = 全部因子數）。
3. 候選（M 外且 |t| > limit）；移除候選（M 內且 |t| ≤ limit）。
4. `direction`：`backward` / `forward` / `both`（預設）。先移除最不顯著者；否則加入最顯著者（`add="best"`）或全部候選（`add="all"`）。
5. 重新配適並重複，直到收斂 / `max_steps` / 加入會使 df < `min_df` / 模型循環。

預設從**全模型**開始（需 df ≥ `min_df`），因為從空模型開始，σ 會被待找的真實效應灌大。每次迭代保存：模型內容、df、σ、t-limit、Bonferroni limit、各因子 t、候選、移除候選、加入/移除/保留、停止原因（`IterationRecord`）。

### 2.3 Lenth（無誤差 df 時）

m 個 contrast `cⱼ`：`s0 = 1.5·median|c|`，`PSE = 1.5·median{|c| : |c| < 2.5·s0}`，`ME = t_{1−α/2, m/3}·PSE`，`SME = t_{γ, m/3}·PSE`，`γ = (1+(1−α)^{1/m})/2`（Lenth, 1989）。
適用：效應稀疏、contrast 正交且等變異、近似常態。限制：`t_{m/3}` 參考分配是近似；**PSE 不是 OLS 標準誤**；不產生 p 值；limits 僅作指引。`method="auto"` 在 df < 1 時自動退回並警告。

### 2.4 Half-Normal Plot

遞增排序 |effect|，位置 `pᵢ = 0.5 + 0.5(i−0.5)/m`，分位數 `qᵢ = Φ⁻¹(pᵢ)`（即 |Z| 的分位數）。圖：X = qᵢ，Y = |effect|（`style="probability"` 為 Design-Expert 式座標軸）。**參考線與篩選分開計算**：對最小 50 %（`fraction`）的點做過原點最小平方，斜率 ≈ 雜訊尺度。PB 未指派的 dummy columns 以空心灰點畫出，可作為誤差的視覺參考。

### 2.5 圖表尺度

`plot_pareto(scale="auto")`：有誤差 df → Y = |t|，threshold = t-limit、Bonferroni limit；無 df → Y = |Effect|，threshold = Lenth ME / SME。`scale="effect"` 時 limit 轉為 `limit × SE(Effect)`（SE 不均一時不轉換）。永遠不會把不同尺度畫在同一張圖。粗框 = 已選入；橘 = 正向、藍 = 負向。

## 3. 混淆（aliasing）與其他限制

* PB 只用於**主效應篩選**。N = 2ᵐ（8、16、32…）：主效應與部分二因子交互**完全**混淆；N = 12、20、24…：交互作用**部分**混淆於多個主效應（N = 12 時相關係數 ±1/3，見測試），一個交互作用可偏誤多個主效應。交互作用**不可估計**。
* `超過 limit` ≠ `確定為關鍵因子`。篩選結果是**候補清單**：逐步選擇用同一份資料選模型再檢定，選後 p 值 / CI 偏樂觀，也不保證因果。請以後續實驗（RSM 階段）確認。
* **實測行為（seed 固定，300 組模擬資料，PB-12 + 4 center points，8 個因子）**：8 個全無效因子時，Bonferroni 規則至少誤標 1 個的比例約 6–7 %（名目 5 %）；僅用 t-limit 約 39–45 %（即使不做逐步篩選，8 個因子的 1−0.95⁸ 也已是 34 %）。因此 t-limit 規則適合「寬鬆找候補」，要控制誤報請用 `criterion="bonferroni"`。測試以區間斷言這些行為。
* PB 大小：由 Sylvester、Paley I / II、Kronecker 建構並驗證 `HHᵀ = NI`（4–96 之間除 52、92 外皆可）；N = 12、20、24、44 採 Paley I 型建構（文獻上與經典循環式 PB 屬同一類）；**本專案未將這些矩陣與經典 PB 生成列逐項比對**，只驗證了正交性、平衡性與 −1/+1 水準。
* 類別型因子只有 ±1 兩水準，不支援 center points。


## 4. Fractional Factorial（`screening/fractional.py`）

**`fractional_factorial(factors, p=0, generators=None, center_points=0)`**：基底因子（前 k−p 個）為標準順序（Yates，第一個因子變動最快）的完整 2^(k−p)，其餘因子由生成器（如 `"E=ABCD"`）產生。**定義關係、解析度與別名結構皆由生成器實際計算**（`info["defining_relation"]`、`info["resolution"]`、`alias_table(k, generators)`），不是查表。常用 (k, p) 內建標準生成器，其解析度有測試對照文獻值（例如 2⁴⁻¹ᵢᵥ、2⁵⁻¹ᵥ、2⁶⁻²ᵢᵥ、2⁸⁻²ᵥ）。

其餘互不別名的對比欄位（以最低階交互作用命名，如 `AB`、`ABC`）存為 `dummy` 欄位（`info["dummy_kind"]="interaction"`）：用於 Lenth / half-normal（Daniel plot，圖例標示「Interaction contrast」），**同時被併入誤差自由度**。若交互作用真實存在，主效應模型會高估 σ（t 值偏小）——Montgomery 2⁴ 範例即是（AC、AD 很大，A 的 t 僅 2.83，df = 11）。要估計交互作用請進入 RSM 階段。

## 5. 驗證範圍

* 測試：Core 資料結構、PB / FF 的結構與正交性、統計量與獨立公式 / SciPy 對照、圖表資料、多 seed 回歸。
* **教科書對照**：Montgomery《Design and Analysis of Experiments》2⁴ 過濾速率範例。資料與已知效應（A=21.625、B=3.125、C=9.875、D=14.625、AC=−18.125、AD=16.625 …）憑記憶輸入，程式**完全重現**這些效應；請以你手邊的教科書再核對一次資料表。**尚未與 R（FrF2 / rsm）做對照**（本環境無 R、無網路）。
* 未做任何商業軟體相容聲明；先前 PDF 的 limit 數值（df=10、m=19 → 2.22814 / 3.97276）已由 SciPy 重現，非 Design-Expert 原始輸出對照。
* 依賴決策（ADR-008）：本階段**自行實作** PB / FF（沙箱無 pyDOE3 且無網路）；OLS 以 NumPy 實作，statsmodels 僅作選用交叉驗證。
