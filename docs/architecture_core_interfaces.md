# Phase 1：核心介面（定案）

> 這份文件是 `architecture.md` §5「核心資料契約」在 Phase 1 的定案內容，請併入 repo 的 `architecture.md`。

## 決定的事項（Roadmap §5 的待決項目）

| 項目 | 決定 |
|---|---|
| 座標編碼 | coded ∈ [−1, +1]：`coded = (x − (high+low)/2) / ((high−low)/2)`；中心點 = 0。類別型因子只有 ±1。設計若需超出區間（如外接 CCD），資料集 `info["allow_outside"]=True` 才允許 |
| `ExperimentDataset` 欄位 | 下表 |
| Screening 實作 | 自行實作 PB / FF（ADR-008：沙箱無 pyDOE3、無網路）；OLS 以 NumPy 實作 |

## 契約

| 物件 | 欄位 / 行為 |
|---|---|
| `Factor(name, low, high, unit, factor_type, metadata)` | `to_coded` / `from_coded`；`low < high`、有限、名稱非空；`numeric` 或 `categorical`（兩水準） |
| `Response(name, weight, r2_threshold, direction, target, unit)` | 驗證：`weight ≥ 0`、`r2_threshold ≤ 1`、`direction ∈ {None, maximize, minimize, target}` |
| `ExperimentDataset` | 一個 coded 矩陣 `X`（n×k）**由所有響應共用**；每個響應一欄（NaN = 尚未量測）；`std_order`（= `run_id`，列識別，不變）與 `run_order`（執行順序，排列）分離；`point_type`、`source`（`initial` / `AL-iter-k` / `validation` …）、`stage`；`info`（design_type、seed、模型規格…）；`response_specs`（各響應的 `Response`） |
| 操作 | `add_response(name, values, order="std"\|"run")`、`append_runs(coded, source, values)`（新 run 排在既有 run 之後）、`subset` / `copy`、`training()` / `validation()`、`replicate_groups()` / `n_unique_points` / `n_replicated_runs`、`validate()`、`to_csv` / `from_csv`、`randomize(seed)`、`experiment(run_id) -> Experiment` |
| `Experiment` | 單一 run 的唯讀檢視（coded、actual、responses、point_type、source、stage） |

## 不變條件（有測試）

* 所有響應與同一個 X 對齊；長度不符的響應會被拒絕。
* `append_runs` 後既有響應補 NaN、`run_order` 延續、`source` / `stage` 記錄新 run 的來源。
* CSV 往返無損（含 source、stage、response 設定、執行順序、dummy 欄位）；actual 與 coded 欄位不一致時報錯。
* 驗證資料（`source="validation"`）可由 `training()` 排除，不會混入訓練。
* 所有隨機程序（`randomize`、設計產生）接受顯式 seed 並記錄於 `info["seed"]`。

## Phase 2 補充：seed 管理與設計資料的交接欄位

| 項目 | 決定 |
|---|---|
| Seed 規則 | 任何隨機程序都接受 `seed`（非負整數或 None）；`None` 會產生並**記錄**種子；不使用 NumPy 全域亂數 |
| `core/seed.py` | `check_seed`、`resolve_seed`、`make_rng`、`derive_seed(root, *names)`（SHA-256，跨行程一致）、`SeedManager`（root＋具名獨立子串流，`as_dict()` 供記錄） |
| 記錄位置 | `dataset.info["seed"]`（root）與 `dataset.info["seeds"]`（root＋實際使用的子 seed） |
| 設計資料交接（給 RSM 建模） | 設計函式回傳 `ExperimentDataset`；`info` 含 `design_type`、`model_spec`（`ModelSpec.to_dict()`，可用 `ModelSpec.from_dict` 還原）、`rank`、`cond_X`、`I_criterion` / `logdet_XtX`、`seeds`；`point_type` ∈ {factorial, axial, center, optimal, included, augment}；`source` 記錄 augmentation 來源 |
| 新例外 | `InfeasibleRegionError(DesignError)`：約束區域為空或可行比例過小 |

## Phase 3 補充：Active Learning 與資料集的交接

| 項目 | 決定 |
|---|---|
| 新增 run 的標記 | `source = "AL-iter-<round>"`、`point_type = "active_learning"`、`stage` 遞增；驗證資料用 `source="validation"`，不算訓練資料也不算預算 |
| 預算計數 | `n_runs` = 訓練 run 數（不含 validation）；另有 `n_unique_points` 與 `n_replicated_runs`；`N_max` 永不超過（最後一批會被縮減） |
| 歷史紀錄 | `ActiveLearner.history`（每輪：run 數、各響應預測 R² 與狀態、聚合 R²、停止決策、超參數、GPR 擬合資訊、CV 設定、評估集指標、建議點）與 `history_df()` |
| 獨立性 | `active_learning` 只依賴 `core`、`design`；不得 import `rsm`、`residual`（ADR-005，測試強制） |

## Phase 4/5 補充：RSM 與殘差修正的交接

| 項目 | 決定 |
|---|---|
| RSM 輸入 | 設計資料集（`ExperimentDataset`）：`rsm.fit(ds, response, ...)` 只用 `training_mask` 內、該響應已量測的 run；validation 永不參與 |
| `RSMResult` | `terms`（指數 tuple）、`beta`、`se`/`t`/`p`/`ci`、`vif`、`leverage`、`resid`、`predict(Z, return_se)`、`equation(coded)`、`anova()`、`diagnostics()`、`selection`（選項紀錄）、`factors`、`run_order` |
| 殘差修正輸入 | 同一份資料集；`residual.fit(ds, response, ...)` 回傳 `HybridModel`（`base: RSMResult`、`learner`、`lam`、`report`、`oof`、`config`），`predict` / `predict_rsm` / `evaluate` |
| 模組依賴 | `rsm`、`diagnostics` 只依賴 `core`、`design`；`residual` 依賴 `core`、`rsm`；`active_learning` 與 `residual` 互不 import（ADR-005，測試強制） |
| 隨機性 | OOF 折、學習器與內層 CV 的 seed 由 `SeedManager` 分流並記錄在 `HybridModel.config["seeds"]` |

