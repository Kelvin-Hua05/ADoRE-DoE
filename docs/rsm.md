# Phase 4：最終 RSM 建模與診斷

範圍：多項式（一階 / 含交互作用 / 二階）OLS、係數表（SE、CI、VIF）、模型統計量、ANOVA（Type III）與 Lack-of-Fit、Stepwise / AICc / BIC 選項、hierarchy 約束、明確方程式（coded 與實際單位）、殘差與影響力診斷。
**不做**（Roadmap §4 延後）：資料轉換、WLS、Box-Cox。

## 1. API

```python
import adore_doe as ad
from adore_doe import rsm

res = rsm.fit(ds, "Capacitance", model="quadratic")                       # 訓練 run；validation 不參與
res = rsm.fit(ds, "Capacitance", model="quadratic",
              selection={"method": "stepwise", "criterion": "p", "alpha_in": 0.10, "alpha_out": 0.10,
                         "hierarchy": "strong"})                         # 先選項再擬合
res.coef_table(); res.fit_statistics(); res.anova()
print(res.equation(coded=True)); print(res.equation(coded=False))        # coded / 實際單位
res.predict(Z, return_se=True)
d = res.diagnostics(); d.table; d.summary(); d.plot_normal(); d.plot_residual_vs_predicted()
```

## 2. 公式（coded 因子；p = 項數，n = run 數）

| 量 | 定義 |
|---|---|
| OLS | `β̂ = argmin‖y − Xβ‖²`（QR），`σ² = RSS/(n−p)`，`Cov(β̂) = σ²(X'X)⁻¹`，`t = b/SE`，`CI = b ± t₁₋α/₂,ₙ₋ₚ SE` |
| Std. Dev. / Mean / C.V.% | `√MSE` / `ȳ` / `100·Std.Dev/ȳ` |
| R² / Adj R² | `1 − RSS/SST` / `1 − [RSS/(n−p)]/[SST/(n−1)]` |
| PRESS / Predicted R² | `Σ(eᵢ/(1−hᵢᵢ))²`（精確的留一法）/ `1 − PRESS/SST` |
| Adeq Precision | `(max ŷ − min ŷ)/√(pσ²/n)`（設計點上） |
| VIF | 非常數欄相關矩陣之逆的對角線 |
| AIC / AICc / BIC | `n ln(RSS/n) + 2p` / `+ 2p(p+1)/(n−p−1)` / `n ln(RSS/n) + p ln n`（同一資料下各模型共有的常數略去；`aic_sm`、`bic_sm` 補上 `n(1+ln2π)`，等於 statsmodels 的數值） |

**誠實處理的情況**：模型不可識別 → `RankDeficientError`（列出被混淆的項，不用偽逆；`on_rank_deficient="drop"` 才會刪項並記錄）；殘差自由度為 0 → SE / t / p / CI / 預測 R² 為 NaN 並警告；槓桿值 = 1 → 學生化殘差與 Cook's D 為 NaN 並說明。

## 3. ANOVA 與 Lack-of-Fit

Model / 各項採 **Type III（partial）平方和**：`SSⱼ = RSS(去掉 j) − RSS(完整)`，`F = SS/MSE`（1 個 df 時等於係數 t 的平方，p 值相同）。
Lack-of-Fit 需要**純誤差**：`SS_PE = Σ_g Σ_{i∈g}(yᵢ − ȳ_g)²`，`df_PE = n − G`（G = 不同設定數），`SS_LOF = RSS − SS_PE`，`df_LOF = (n−p) − df_PE`。
**沒有重複點（例如 ADR-010 的 CCD 只有 1 個中心點）或 `df_LOF = 0` 時，LOF 回報「NOT AVAILABLE」與原因，不會硬算。**

## 4. 選項（`rsm.select_terms`）

| 準則 | 規則 |
|---|---|
| `p`（預設；α_in = α_out = 0.10） | 進入：p 值最小且 < α_in；移除：p 值最大且 > α_out |
| `aicc` / `bic` / `aic` | 只在準則值下降（> 1e-9）時移動 |

`method`：`stepwise` / `forward` / `backward`。**`hierarchy="strong"`（預設）**：項與其缺少的 parents 一起進入（候選項在「加入 parents 後的模型」中評估；p 準則用候選項自己的 p 值），且有 child 存在時不可移除 parent；`"none"` 不設限。可用 `force` 保留指定項。每一步都記錄在 `SelectionResult.history`。
這是**貪婪搜尋**，不保證最佳子集；選模後的 p 值與信賴區間偏樂觀（選擇與檢定用了同一份資料）。

## 5. 診斷

`hᵢᵢ`（hat 對角線）、內部學生化 `rᵢ = eᵢ/(σ√(1−hᵢᵢ))`、外部學生化 `tᵢ = rᵢ√((n−p−1)/(n−p−rᵢ²))`、Cook's D `= rᵢ²h/(p(1−h))`、DFFITS。旗標：outlier（`tᵢ` 的 Bonferroni 調整 p < 0.05）、高槓桿（h > 2p/n）、influential（D > 1）。常態性：外部學生化殘差的 Shapiro–Wilk；變異數：|殘差| 與預測值的 Spearman 相關。圖：Normal plot、Residual vs Predicted / Run / Factor、Predicted vs Actual（需要 `[plot]`）。
**Shapiro–Wilk 套用在學生化殘差上只是近似**：模擬（n = 40、p = 10、高斯誤差）的誤報率約 10%（名目 5%），對指數誤差的檢出力約 75%。請把它當作警示，不要當作精確的檢定。

## 6. 驗證範圍（誠實聲明）

* **教科書對照（Montgomery, *Design and Analysis of Experiments*, RSM 章節的化學製程範例）**：資料憑記憶輸入。二階模型的四個因子點的排列，我以窮舉 576 種排列比對：**只有一種排列**能同時重現書中六個係數（79.94、0.995、0.515、−1.376、−1.001、0.25）。程式算出 79.94、0.9951、0.5152、0.25、−1.3764、−1.0013；純誤差 SS = 0.2120（df 4）、LOF 不顯著；第一階含中心點的範例：係數 40.44 / 0.775 / 0.325、純誤差 SS = 0.1720、A 與 B 的 SS = 2.4025 / 0.4225。**請對照你手邊的原書再核對一次資料表。**
* 與獨立的 NumPy / SciPy 公式對照：係數、SE、t、p、CI、R²、PRESS（與逐筆留一重擬合）、VIF、診斷量（外部學生化殘差與逐筆重擬合）。
* LOF 檢定的校準：3000 次模擬，名目 5% / 10% / 50% 的拒絕率為 4.6% / 10.1% / 49.6%。
* **statsmodels 交叉驗證**：測試已寫好（係數、SE、p、CI、R²、AIC/BIC、hat、PRESS、學生化殘差、Cook's D、DFFITS），**在安裝了 statsmodels 的環境才會執行**。開發沙箱沒有 statsmodels，所以這些測試我**尚未實際執行過**。
* **未與 R 的 `rsm` 對照**（沙箱無 R、無網路），因此不宣稱與 R 一致。
* 限制：二階以內的多項式（`parents` 只定義到二階項）；Type III 平方和在存在高階項時與「邊際」解讀不同；Stepwise 為貪婪搜尋。
