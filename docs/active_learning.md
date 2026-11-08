# Phase 3：Module 1 Active Learning

範圍：GPR 代理模型、折內不洩漏的交叉驗證預測 R²、多響應停止條件、固定評估集、acquisition（I-optimal augmentation、IVR、Max Variance、Random）、批次選點、`ActiveLearner` 迴圈。
**不依賴** `rsm` 與 `residual`（ADR-005，有 import 檢查測試）。**不做**：Expected Improvement / 最佳化導向策略、GA、離散變數、非 GPR 代理模型（Roadmap §4 延後項目）。

## 1. API

```python
import adore_doe as ad
from adore_doe import design, active_learning as al

factors = [ad.Factor("Width", 10, 50), ad.Factor("Spacing", 5, 30), ad.Factor("Thickness", 20, 80)]
ds = design.i_optimal(factors, "quadratic", runs=12, seed=1)           # 初始設計
ds.add_response("C", c_values); ds.add_response("R", r_values)         # 使用者自己的模擬結果

stop = al.MultiResponseStopping(n_max=30, response_min_r2=0.90, aggregate_r2=0.93, batch_size=2)
L = al.ActiveLearner(ds, ["C", "R"], stop, strategy="ivr", seed=7)

while True:
    decision = L.assess()                     # 擬合 GPR、CV 預測 R²、檢查停止條件
    if decision.stop: break
    sel = L.recommend()                       # 只「建議」：不執行模擬、不改資料集
    sel.to_csv("next_runs.csv", factors)
    L.update(sel.points, run_my_simulation(sel.points))   # 使用者提供結果 {"C": [...], "R": [...]}
print(L.summary()); L.history_df()
```
`L.run(simulator)` 把上面的迴圈包起來（合成函數 / benchmark 用）。

## 2. 各元件與對應的決定

| 元件 | 內容 | 依據 |
|---|---|---|
| `GaussianProcess` | 每個響應一個 GPR；Matérn 5/2 + ARD（`rbf` 可選）；y 內部標準化、輸出回原單位；超參數 `log[ℓ₁..ℓ_d, σ_f², σ_n²]` 以多起點 L-BFGS-B 在**明確邊界**內最大化邊際似然（起點：warm start、預設、`n_restarts` 個隨機）；失敗時退回預設超參數並警告、`fit_info` 記錄；噪音可學習或固定；所有隨機由 seed 控制 | ADR-002 |
| 每輪流程 | 每輪重估超參數，以上一輪結果 warm start；**同一輪內**評估候選點時超參數固定，用 rank-1 更新 | ADR-002 |
| 預測 R² | `R² = 1 − SSE/SST`；SSE 為池化 out-of-fold 預測，SST = Σ(y−ȳ)²；**負值保留**；SST=0、樣本不足、預測非有限 → `NaN` + 明確 `status`，視為未達標；訓練 R² 不用於停止 | ADR-009 |
| 交叉驗證 | `kfold`（預設 5 折）、`repeated_kfold`、`loo`；每折**含超參數**都只用該折訓練資料重新擬合 | ADR-009 |
| 停止條件 | 每個響應 `R²ₘ ≥ τₘ`（預設 0.95，可逐響應設定）**且**（若設定）聚合 R²（平均或加權平均，權重非負且和為 1）≥ τ_agg；聚合**不能取代**單一門檻；`n_runs ≥ N_min`（預設 k(k−1)/2）且**連續 3 輪**達標 → `target_reached`；否則 `n_runs ≥ N_max` → `budget_exhausted`（目標**未達成**）。`N_max` 無預設值 | ADR-009、§6.2 |
| 評估集 | 固定、與訓練資料獨立（`check_disjoint`）、可由 seed 重現；有真值時回報真實 RMSE/MAE/R²，**沒有真值時只回報平均預測變異**，不捏造真值 | §5.3 |
| 候選點 | 由 `design.generate_candidates` 產生（Sobol/LHS/grid/user），排除既有 run，支援最小距離與線性約束；**產生候選點不執行任何模擬** | ADR-001 |

## 3. Acquisition

| strategy | 定義 | 依賴 |
|---|---|---|
| `i_optimal` | **主策略（ADR-003）**：對指定多項式模型（預設 quadratic）做 I-optimal augmentation，選使 `tr[(X'X)⁻¹B]` 下降最多的候選點（Sherman–Morrison）。**只依賴設計點，與響應值、與 GPR 變異無關** | 既有 run + 模型 |
| `ivr` | 整合 GPR 變異下降：`ΔIV(c) = (1/N)Σᵤ K(u,c)² / (K(c,c)+σ_n²)`，u 為區域內積分點 | 每個響應的 GPR |
| `max_variance` | 最大 GPR 預測變異（**baseline**） | 每個響應的 GPR |
| `random` | 均勻隨機候選點（**baseline**） | — |

**多響應（GPR 類策略）**：每個響應的變異先除以該響應的資料變異數（ADR-002），再以加權平均（預設等權）或最大值（`aggregate="max"`，最差響應）合併；因此單位不同的響應不會互相壓過（有測試：把一個響應乘以 10⁶ 再加偏移，選點不變）。

**批次**：貪婪選取，在固定超參數下是**精確的**——GP 的潛在後驗共變異數不依賴尚未得知的 y，所以選定 c 後每個響應都以 `K ← K − K[:,c]K[c,:]/(K[c,c]+σ_n²)` 更新；I-optimal 同理。另可設最小距離（對既有 run 與批次內已選點），避免聚集。
測試以「每選一點就用固定超參數重新擬合、逐一嘗試所有候選點」的暴力結果驗證：IVR、Max Variance、I-optimal 的選點與分數都完全一致。

**不可混為一談的三個量**：①I-optimal 的設計變異（多項式模型，σ² f'(X'X)⁻¹f）；②GPR 的預測變異（kernel 與超參數決定）；③各響應的預測準確度（CV R²）。IVR **不是** I-optimality（research_methodology §4.3）。

## 4. 策略比較實驗（驗收項目；合成資料）

`examples/phase3_strategy_comparison.py`：3 個因子、3 個已知解析函數（**Synthetic**），初始 12 點 Latin hypercube（每個 seed 不同），再單點序貫加 12 個 run（共 24），每輪重擬 GPR；指標為固定 1000 點評估集上的**真實** RMSE ÷ 真值標準差（NRMSE）；8 個 seed。最終 NRMSE（平均 ± 標準差）：

| 函數 | random | max_variance | ivr | i_optimal |
|---|---|---|---|---|
| smooth（二次＋交互作用） | 0.0319 ± 0.0102 | 0.0111 ± 0.0013 | 0.0117 ± 0.0016 | 0.0110 ± 0.0011 |
| nonlinear（sin/cos） | 0.3189 ± 0.0905 | 0.3385 ± 0.1107 | 0.3039 ± 0.1038 | 0.3303 ± 0.0960 |
| localized（高斯峰） | **0.5992 ± 0.1444** | 0.8462 ± 0.1201 | 0.6737 ± 0.2032 | 0.8814 ± 0.0868 |

配對比較（同一 seed；勝場＝最終 NRMSE 較低）：
* smooth：ivr、i_optimal、max_variance 對 random 皆 8/8 勝；三者彼此幾乎打平（差異 < 0.001）。
* nonlinear：沒有明確贏家——ivr 對 random 平均好 0.015（4/8 勝）、i_optimal 對 random 平均差 0.011；差異都小於 seed 間標準差（約 0.1），**不能判定優劣**。
* localized：**random 平均最好**；max_variance 對 random 只有 1/8 勝、i_optimal 0/8 勝；ivr 對 random 5/8 勝但平均較差（−0.075），對 max_variance 則 7/8 勝（平均 +0.17）。

**結論（如實記錄）**：Roadmap 驗收的「至少一種全域導向策略同時勝過隨機與 Max Variance」在本實驗**沒有被證明**。IVR 對 Max Variance：smooth 打平（差 −0.0006）、nonlinear 平均較好但只有 4/8 勝（在雜訊內）、localized 明顯較好（7/8 勝）；但 IVR 不能穩定勝過隨機；I-optimal augmentation 在平滑函數上與其他策略打平，在局部峰函數上明顯較差。
可能原因（**尚未驗證**）：Max Variance 與多項式 I-optimal 偏好邊界／角落的點，而局部峰在區域內部；24 個點、單點序貫、8 個 seed 的統計力有限；GPR 超參數每輪重估也會影響結果。這些結果**只適用於這 3 個函數與這個設定**，不應推廣；更完整的比較屬於 Phase 6（Benchmark）。

## 5. Seed 與可重現性

`ActiveLearner(seed=…)` 以 SeedManager 分出子串流（`gpr-<響應>-<輪>`、`cv-<響應>-<輪>`、`candidates-<輪>`、`select-<輪>`…），同 seed 的整個迴圈完全相同（有測試）；實際使用的 root seed 在 `summary()["seed"]`。

## 6. 驗證範圍與限制（誠實聲明）

已驗證（pytest）：GPR 的核、插值、變異、後驗共變異數 PSD、**rank-1 更新等於固定超參數重擬合**、ARD 辨識無關維度；CV 折的切分性質與**構造性的無洩漏測試**（改變 yᵢ 不會改變 i 自己的 out-of-fold 預測；固定超參數的選項會洩漏並被標示）；R² 的定義、負值、未定義狀態；停止規則（含 [0.96, 0.95, 0.89] 必須繼續、[0.96, 0.95, 0.94] 平均剛好 0.95 即達標、聚合不能取代單一門檻、budget_exhausted 不當成功）；acquisition 與暴力法一致；預算不超過 N_max；同 seed 可重現；不 import rsm/residual。
**限制**：
* 預測 R² 以全部 y 的平均為 SST（池化 OOF 的標準做法），樣本很少時估計噪音大；「連續 3 輪」降低但不消除誤判。
* CV 的 `refit_hyperparameters=False` 較快但會洩漏，預設為 True。
* GPR 超參數受邊界限制（長度尺度 0.05–20 coded 單位等），極端函數可能撞到邊界；失敗時退回預設並警告。
* 沒有與 scikit-learn 或其他 GPR 實作做數值對照（自行實作，僅以上述性質與暴力法驗證）。
* 候選點數量有限（預設 500），選點只在候選集內最佳。
* 離散變數、Expected Improvement、非 GPR 代理模型未實作。
