# 演算法與技術決策紀錄（ADR）

狀態說明：**Accepted** = 已確認；**Proposed** = 待驗證或待確認。

最後更新：2026-11-04。符號： $k$ = 因子數（輸入變數個數）。

* * *

## ADR-001：I-optimal 整合進 RSM Design 與 Active Learning，而非獨立階段

**狀態**：Accepted

- **Context**：舊規劃以 D-optimal 為主；I-optimal 準則直接對應「全域預測變異」，與本研究目標一致。
- **Decision**：
  1. I-optimal 為主要 optimal 準則，整合在 RSM Design、Design Augmentation、Sequential Design；D-optimal 保留作比較。
  2. **積分點數 $m$ 為可設定參數**，不固定單一數字。以準則值隨 $m$ 增加（例如逐次加倍）的相對變化作收斂檢查，變化小於容許值（預設 1%，可調）即視為近似足夠。
  3. **候選集約束**：第一版支援 bound constraints（各因子上下限）與 linear constraints（線性不等式）；非線性約束延後。積分點與候選點以準擬隨機序列（如 Sobol）產生，並以 rejection sampling 排除不滿足線性約束的點。
- **Alternatives**：(1) 獨立的 I-optimal 開發階段；(2) 繼續以 D-optimal 為主；(3) 固定 $m$。
- **Rationale**：I-optimal 不是獨立工作流，而是同一套「模型 + 區域 + 候選集」機制的準則選項；固定 $m$ 在因子數增加時會影響計算品質。
- **Consequences**：`design` 需要共用的候選集 / 積分點 / 秩檢查機制；Module 1 可呼叫 `design.augmentation`。約束越嚴格，rejection sampling 的接受率越低，需記錄接受率。
- **Open questions**：收斂容許值的預設是否適用於高維；線性約束區域極小時的取樣效率。

## ADR-002：GPR 作為 Module 1 的初始代理模型

**狀態**：Accepted

- **Context**：初始樣本少（約 10–30 runs），需要可給出預測不確定度的代理模型。
- **Decision**：
  1. 以 GPR 為初始 surrogate，**每個響應各一個**，各自擁有獨立的 kernel 與超參數。
  2. **Kernel**：Matérn 5/2 + ARD；RBF 保留作為比較選項。
  3. **超參數更新**：每輪新增 run 後重新最佳化，並以上一輪結果 warm-start。同一輪內評估候選點時固定超參數，以便使用 rank-1 更新。
  4. **多響應聚合**：將各響應的預測變異正規化（除以該響應訓練資料的變異數）後取平均，作為共同的 acquisition 分數；以 max 聚合作為比較選項。
- **Alternatives**：Random Forest（不確定度估計較粗）、ANN / ensemble（小樣本不穩定）、貝氏多項式回歸；固定超參數；共享 kernel。
- **Rationale**：GPR 在小樣本下給出校準的後驗變異；ARD 讓模型學習各因子的尺度差異；資料量小，初期優先模型適應性而非更新速度；獨立建模容易實作，也方便單一響應與多響應比較。
- **Consequences**：結果依賴 kernel 與超參數假設；`Surrogate` 介面保持可替換。超參數每輪重估在小樣本下可能不穩，需設定超參數範圍並記錄最佳化失敗的處理方式。
- **Open questions**：超參數的上下界；正規化方式與研究方法論 §6.1 的對應；平均聚合是否會掩蓋單一響應的高不確定度。

## ADR-003：Active Learning 的目標是全域預測改善

**狀態**：Accepted

- **Context**：單純挑最大預測變異的點傾向集中在邊界 / 外圍，對全域預測的改善有限。
- **Decision**：
  1. **選點策略（acquisition）**：以 I-optimal augmentation 為主策略；Max Variance 僅作 baseline；Expected Global Improvement 列為進階研究方向，目前不實作。
  2. 策略以評估集上的全域指標（Global RMSE / MAE / predictive R²、平均預測變異）比較。
  3. **指標分工**：預測 R² 為主要停止指標（見 ADR-009）；Global RMSE / MAE 僅作報告與策略比較，不參與停止判斷。
- **Alternatives**：以單點不確定度為準；以 Expected Global Improvement 為主。
- **Rationale**：研究目標是整個設計區域的預測準確度。R² 無單位，可跨響應設同一門檻；RMSE / MAE 有單位，不同響應間不可直接比較。
- **Consequences**：需要固定評估集與（模擬器 / 資料池情境下的）真值；真實實驗下只能用模型不確定度與 CV 代理，須分開報告。
- **Open questions**：無（多響應聚合已於 ADR-002 決定）。

## ADR-004：GA 保留給最終最佳化

**狀態**：Accepted

- **Context**：舊規劃把 GA 列為流程步驟之一；Active Learning 的問題是「下一個實驗做哪裡」，不是黑箱全域搜尋。
- **Decision**：GA / NSGA-II / SLSQP / Desirability 只用於最終製程 / 設計參數最佳化，不進入 Module 1。
- **Alternatives**：以 GA 搜尋 acquisition 最大值。
- **Rationale**：候選集上的 acquisition 評估已足夠，不需要 GA；避免把最佳化與資料蒐集混在一起。
- **Consequences**：`optimize` 在 Phase 7 才開發，只作用於已驗證的最終模型。
- **Open questions**：若候選集過大導致 acquisition 評估成本高，才重新評估（需明確研究理由）。

## ADR-005：Module 1 與 Module 2 彼此獨立

**狀態**：Accepted

- **Context**：Active Learning 必須能在尚無最終 RSM 時運作；殘差修正需要能獨立套用於任何提供的 RSM 與資料。
- **Decision**：二者只透過 `ExperimentDataset` 交接；`active_learning` 不得依賴 `rsm` / `residual`；Module 2 不是 Module 1 的前置條件。
- **Alternatives**：以最終 RSM 的指標驅動 Active Learning。
- **Rationale**：降低耦合，可單獨驗證與單獨使用。
- **Consequences**：GPR 的停止指標與最終 RSM 的表現可能不一致，須在 Benchmark 階段兩者並報（見 ADR-012）。
- **Open questions**：已移至 ADR-012。

## ADR-006：殘差修正必須使用 OOF 殘差

**狀態**：Accepted

- **Context**：以訓練集上的 in-sample 殘差訓練殘差模型，殘差被低估，殘差模型學到的是擬合假象。
- **Decision**：
  1. 殘差目標由 OOF RSM 預測建立（折內重做模型選擇），殘差模型以 OOF 殘差訓練，並在獨立驗證資料評估。
  2. **OOF 產生方式**：第一版使用 Repeated 5-fold CV；LOO 作為小樣本情境的比較方法。
  3. **Shrinkage**：不預設固定。修正式為 $Y_{\text{corrected}} = Y_{\text{RSM}} + \lambda\,\hat{e}_{ML}$， $0 \le \lambda \le 1$。Benchmark 比較無 shrinkage($\lambda = 1$)與有 shrinkage($\lambda$ 以 OOF 選擇)的差異。
- **Alternatives**：in-sample 殘差（洩漏）；單次 hold-out 切分（小樣本浪費資料）；固定使用 shrinkage。
- **Rationale**：避免洩漏，估計真實泛化表現；Repeated K-fold 能觀察結果穩定性，且計算成本低於 LOO。
- **Consequences**：OOF 殘差略大於全資料 RSM 殘差（分布偏差）；小樣本下 CV 方差大，須文件化；以 OOF 選 $\lambda$ 時須避免用同一份資料再評估（需巢狀或獨立驗證）。
- **Open questions**：重複次數預設值； $\lambda$ 選擇是否需要巢狀 CV。

## ADR-008：依賴策略（最小化、延後決定）與 BBD 移除

**狀態**：Accepted

- **Context**：舊規劃列出 `statsmodels`、`scikit-learn`、`pyDOE3`、`SALib`、`pymoo`、`DEAP`；其中 `pymoo` 與 `DEAP` 功能重疊。
- **Decision**：
  1. 目前 `pyproject.toml` 只宣告 `numpy`、`scipy`、`pandas`，以及可選的 `plot`（matplotlib）、`ml`（scikit-learn）、`stats`（statsmodels）、`dev`。其餘依功能階段才決定。
  2. 決策原則：**以方便維護與方便控制為主**。
  3. Phase 1/2 時決定 Screening / CCD 用 `pyDOE3` 或自行實作，以及 RSM 建模用 `statsmodels` 或 numpy；Phase 7 時在 `pymoo` 與 `DEAP` 間擇一。
  4. **Box-Behnken Design（BBD）不實作**，從規劃與依賴考量中移除。
- **Alternatives**：先全部裝入。
- **Rationale**：不為尚未實作的功能增加依賴；I-optimal 與 stepwise / LOF 的細節控制多半需要自行實作並對照 R 參考實作。
- **Consequences**：依上述原則，RSM 核心（AICc、hierarchy stepwise）傾向自行實作，並以 `statsmodels` 作對照驗證（待 Phase 1/2 確認）。
- **Phase 1 結果**：Screening（Plackett-Burman、Fractional Factorial）自行實作（開發環境無 pyDOE3 且無網路）；OLS 以 NumPy 實作，statsmodels 僅作選用的交叉驗證。CCD 的實作方式留待 Phase 2 決定。
- **Phase 2 結果**：CCD（依 ADR-010：full factorial、face-centered、中心點預設 1 個）、I-optimal / D-optimal（candidate-exchange 啟發式）與 augmentation 皆自行實作；不使用 pyDOE3。
- **Phase 3 結果**：Active Learning 的 GPR 自行實作（僅用 NumPy / SciPy，不依賴 scikit-learn）；交叉驗證預測 R² 於每折重新最佳化超參數（不 warm-start，避免洩漏）；`active_learning` 不 import `rsm` / `residual`（ADR-005，有測試）。
- **Phase 4/5 結果**：RSM 建模（OLS、ANOVA / LOF、stepwise / AICc / BIC、hierarchy、診斷）以 NumPy / SciPy 自行實作，statsmodels 僅作選用的對照測試；殘差修正的 GP / MLP / SVR 使用 scikit-learn（選用依賴 `ml`），Random Forest 依 ADR-011 延後；`residual` 與 `active_learning` 互不 import（ADR-005，有測試）。
- **Open questions**：RSM 建模最終採用方式（Phase 1/2）。

## ADR-009：預測 R² 的定義與停止條件

**狀態**：Accepted

- **Context**：停止條件依賴 $R^2$；不同定義（訓練、CV、hold-out）與邊界情況會改變結論。
- **Decision**：
  1. **指標**：預測 $R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$，以 held-out / OOF 預測值**池化**後計算（pooled OOF）；負值保留； $\mathrm{SST} = 0$ 或樣本數不足以進行 CV 時，回報原因並視為未達標；訓練 $R^2$ 僅作報告。
  2. **門檻**： $R^2_{\text{threshold}} = 0.95$（可由使用者調整）。
  3. **最小 run 數**： $N_{\min} = k(k-1)/2$， $k$ 為因子數。未達 $N_{\min}$ 前不允許停止。
  4. **最大 run 數**  $N_{\max}$：由使用者設定，無預設值。
  5. **連續達標**：需連續 3 輪預測 $R^2 \ge 0.95$ 才停止。
  6. **多響應**：所有響應都達標才算達標。
  7. **停止規則**：總 run 數 $\ge N_{\min}$ 且連續 3 輪全部響應達標 → 停止；或總 run 數達 $N_{\max}$ → 停止並標記「未達標」；否則繼續選下一點。
- **Alternatives**：Adjusted $R^2$、PRESS-based predicted $R^2$（以 LOO 為特例）；各折平均；單輪達標即停；以平均 $R^2$ 判斷多響應。
- **Rationale**：反映泛化而非擬合；連續達標降低被單輪雜訊誤導的機會；全部響應達標避免某一響應表現差而被平均掉。
- **Consequences**：序貫選點下的 CV 估計可能有偏； $N_{\min} = k(k-1)/2$ 在因子數少時很小（ $k=3$ 為 3、 $k=4$ 為 6、 $k=5$ 為 10），低於二次模型的參數個數 $(k+1)(k+2)/2$，也可能低於初始設計 run 數，此時實際上不會限制停止時機。
- **Open questions**： $N_{\min}$ 是否指「含初始設計的總 run 數」（目前採此解讀），或指初始設計之後新增的 run 數；若採後者，對小 $k$ 才有實質限制效果。

## ADR-010：CCD、I-optimal、D-optimal 在設計比較中的定位

**狀態**：Accepted

- **Context**：移除 BBD 後，需明確定義各設計方法在 benchmark 中的角色，才能公平比較。
- **Decision**：
  1. **CCD**：傳統 RSM benchmark。採 full factorial CCD，軸點 $\alpha = \pm 1$（face-centered，軸點落在各面中心），**中心點 1 個**。Run 數為 $2^k + 2k + 1$（ $k=3$：15； $k=4$：25； $k=5$：43）。
  2. **I-optimal**：主要的 optimal design。
  3. **D-optimal**：比較係數估計表現的 benchmark。
- **Alternatives**：Fractional factorial CCD； $\alpha > 1$（旋轉性 CCD，軸點會超出立方體範圍）。
- **Rationale**： $\alpha = \pm 1$ 使所有點都在因子上下限內，與 bound constraints 一致。
- **Consequences**：中心點只有 1 個，**無重複點可估計純誤差**，因此無法用中心點重複進行 lack-of-fit 檢定；若需要 LOF 須另行設計重複點。Full factorial 的 run 數隨 $k$ 快速增加， $k \ge 5$ 時可能超過 Module 1 的 10–30 run 預算，benchmark 須在相同 run 數預算下說明比較方式。
- **Open questions**：LOF 檢定的純誤差來源（見 Consequences）；各方法的 run 數預算如何對齊。

## ADR-011：殘差修正的候選模型

**狀態**：Accepted

- **Context**：ADR-006 只決定殘差目標的產生方式，尚未決定殘差模型。
- **Decision**：第一版先實作 Gaussian Process、ANN（MLP）、SVR 三種；Random Forest 視研究結果再加入。
- **Alternatives**：一次實作全部模型。
- **Rationale**：GP 給出不確定度，ANN 為研究重點之一，SVR 適合小樣本；分階段避免一開始工作量過大。
- **Consequences**：`Residual` 介面需可替換模型；ANN 在小樣本下需限制網路規模並文件化超參數。
- **Open questions**：ANN 架構與正則化設定。

## ADR-012：Benchmark 如何整合 Module 1 與 Module 2

**狀態**：Accepted

- **Context**：ADR-005 規定兩模組獨立，但未定義最終如何一起比較。GPR 的停止指標與最終 RSM 的表現可能不一致。
- **Decision**：
  1. 以相同 run 數預算比較各設計 / 選點策略（CCD、I-optimal、D-optimal、I-optimal augmentation、Max Variance baseline）。
  2. 在 GPR 停止點，**並報**兩組結果：GPR 的停止指標，以及該 run 數下最終 RSM（及殘差修正後）在固定評估集上的預測表現。
  3. 在 benchmark 層（而非 Module 1 內）提供選項：以 RSM 預測表現作為停止依據，用來與 GPR 停止點比較。
- **Alternatives**：只報 GPR 指標；讓 Module 1 直接依賴 RSM。
- **Rationale**：保持模組獨立，同時量化兩種停止條件的差異。
- **Consequences**：Benchmark 需要固定評估集與模擬器 / 資料池情境的真值；真實實驗只能報 OOF 代理，須分開標示。
- **Open questions**：run 數預算的對齊方式（與 ADR-010 同）。
