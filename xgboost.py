import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import xgboost as xgb
from typing import List, Dict, Tuple
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import json

def build_sliding_window_rfm(
    df: pd.DataFrame,
    date_col: str,
    user_col: str,
    item_col: str,
    amount_col: str,
    observation_days: int = 90,
    label_days: int = 30,
    stride_days: int = 30
) -> pd.DataFrame:
    """
    構建滑動時間視窗 (Sliding Window) 數據集並計算 User-Item 層級的 RFM 特徵
    
    簡述:
        此函式將長條的交易歷史數據轉換為機器學習可用的監督式學習樣本。
        採用「滑動視窗」策略，針對每一個時間切點 (Cutoff Date)，定義一段「觀測期 (Observation Period)」
        用於計算 RFM 特徵，以及一段「預測期 (Label Period)」用於標註是否回購。
        最終將多個視窗的結果合併，形成一個包含時序特徵的訓練資料集。

    Args:
        df (pandas.DataFrame):
            原始交易數據 DataFrame。必須包含日期、用戶 ID、商品 ID 以及金額/數量欄位。
        date_col (str):
            交易日期欄位名稱 (必須為 datetime 格式)。
        user_col (str):
            用戶 ID 欄位名稱。
        item_col (str):
            商品 ID 欄位名稱。
        amount_col (str):
            交易金額或數量欄位名稱 (用於計算 Monetary)。
        observation_days (int, optional):
            觀測期天數 (Lookback Window)。模型將查看過去幾天的數據來生成特徵。
            預設值為 90。
        label_days (int, optional):
            預測期天數 (Prediction Window)。模型將預測未來幾天內是否發生購買。
            預設值為 30。
        stride_days (int, optional):
            視窗滑動的步長 (Stride)。決定每隔多久產生一個新的切點。
            預設值為 30。

    Returns:
        pandas.DataFrame:
            回傳一個整合後的訓練數據集 (Training Dataset)，每一列代表一個 (User, Item, Time_Window) 的樣本。
            主要欄位包括：
            - cut_date (datetime): 該樣本的時間切點 (特徵截止日)
            - {user_col}: 用戶 ID
            - {item_col}: 商品 ID
            - recency (int): 距離觀測期截止日，上次購買該商品經過的天數
            - frequency (int): 觀測期內購買該商品的次數
            - monetary (float): 觀測期內購買該商品的總金額
            - label (int): 目標變數。1 表示在預測期內有回購，0 則無。

    Side effects:
        - 函式執行過程中會印出當前處理的視窗日期範圍進度 (Processing window...)。
        - 若數據量龐大，可能會消耗較多記憶體進行 DataFrame 的合併操作。

    Notes:
        - 為避免資料洩漏 (Data Leakage)，特徵計算嚴格限制在 `[cut_date - observation, cut_date]` 範圍內。
        - 標籤 (Label) 計算嚴格限制在 `(cut_date, cut_date + label]` 範圍內。
        - 僅針對在「觀測期」內有互動的 (User, Item) 產生樣本。若需包含冷門商品 (Zero-inflated)，
          需額外進行 Cross Join 補全 (不在此函式範疇內以節省資源)。
    
    Raises:
        ValueError:
            若必要的欄位名稱不在輸入的 df 中，或 date_col 不是 datetime 類型時拋出。
        
    Example:
        >>> # 假設 transaction_df 是您的交易資料
        >>> train_df = build_sliding_window_rfm(
        ...     transaction_df,
        ...     date_col='transaction_date',
        ...     user_col='customer_id',
        ...     item_col='product_id',
        ...     amount_col='sales_amount',
        ...     observation_days=90, 
        ...     label_days=30
        ... )
        >>> print(train_df.head())
    """
    
    # 1. 基本檢查與前處理
    required_cols = [date_col, user_col, item_col, amount_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"輸入的 DataFrame 缺少必要欄位，請檢查: {required_cols}")
        
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"欄位 '{date_col}' 必須是 datetime 格式。")

    # 確保資料按時間排序，雖非必要但有助於除錯
    df = df.sort_values(date_col)
    
    # 決定視窗滑動的起點與終點
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # 第一個切點：至少要有完整的觀測期資料
    current_cut_date = min_date + timedelta(days=observation_days)
    
    all_windows_data = []
    
    print(f"--- 開始滑動視窗特徵工程 ---")
    print(f"資料範圍: {min_date.date()} 到 {max_date.date()}")
    
    # 2. 開始滑動迴圈
    # 條件：切點加上預測期，不能超過資料的最大日期 (否則無法產生 Label)
    while current_cut_date + timedelta(days=label_days) <= max_date:
        
        # 定義時間範圍
        obs_start = current_cut_date - timedelta(days=observation_days)
        obs_end = current_cut_date
        label_end = current_cut_date + timedelta(days=label_days)
        
        print(f"處理視窗切點: {obs_end.date()} | "
              f"觀測: {obs_start.date()}~{obs_end.date()} | "
              f"預測: {obs_end.date()} (不含)~{label_end.date()}")

        # 3. 切分數據
        # Mask 1: 觀測期數據 (用於 X)
        mask_obs = (df[date_col] > obs_start) & (df[date_col] <= obs_end)
        df_obs = df.loc[mask_obs].copy()
        
        # Mask 2: 預測期數據 (用於 y)
        mask_label = (df[date_col] > obs_end) & (df[date_col] <= label_end)
        df_label = df.loc[mask_label].copy()
        
        if df_obs.empty:
            print(f"  警告: 切點 {obs_end.date()} 的觀測期無數據，跳過此視窗。")
            current_cut_date += timedelta(days=stride_days)
            continue

        # 4. 特徵工程 (Feature Engineering) - 計算 RFM
        # 針對 (User, Item) 分組
        # Recency: 觀測期截止日 - 最近一次購買日 (以天為單位)
        # Frequency: 觀測期內出現次數 (count)
        # Monetary: 觀測期內總金額 (sum)
        
        # 先計算最後購買日期，用於計算 Recency
        last_purchase = df_obs.groupby([user_col, item_col])[date_col].max().reset_index()
        last_purchase['recency'] = (obs_end - last_purchase[date_col]).dt.days
        
        # 計算 Frequency 與 Monetary
        agg_metrics = df_obs.groupby([user_col, item_col]).agg(
            frequency=(date_col, 'count'),
            monetary=(amount_col, 'sum')
        ).reset_index()
        
        # 合併特徵表
        features = pd.merge(last_purchase[[user_col, item_col, 'recency']], 
                            agg_metrics, 
                            on=[user_col, item_col], 
                            how='left')
        
        # 5. 標籤生成 (Label Generation)
        # 目標: 檢查 (User, Item) 是否出現在預測期 (df_label) 中
        # 我們只需要知道 unique 的 (User, Item) 組合
        future_purchases = df_label[[user_col, item_col]].drop_duplicates()
        future_purchases['label'] = 1
        
        # 將 Label 合併回特徵表
        # 使用 left join，若預測期沒出現則補 0
        window_dataset = pd.merge(features, 
                                  future_purchases, 
                                  on=[user_col, item_col], 
                                  how='left')
        window_dataset['label'] = window_dataset['label'].fillna(0).astype(int)
        
        # 6. 加入元數據 (Metadata) 以便追蹤
        window_dataset['cut_date'] = obs_end
        
        all_windows_data.append(window_dataset)
        
        # 7. 滑動到下一個切點
        current_cut_date += timedelta(days=stride_days)
        
    # 8. 合併所有視窗的結果
    if not all_windows_data:
        print("警告: 未生成任何視窗數據。")
        return pd.DataFrame()
        
    final_df = pd.concat(all_windows_data, ignore_index=True)
    
    print(f"--- 完成 ---")
    print(f"總共生成樣本數: {len(final_df)}")
    print(f"標籤分佈 (Label=1): {final_df['label'].mean():.4%}")
    
    return final_df

def compute_average_purchase_interval(orders_df: pd.DataFrame) -> pd.DataFrame:
    """計算每位用戶的平均購買間隔 (days_since_prior_order 的平均值)。"""
    required_cols = {'user_id', 'days_since_prior_order'}
    if not required_cols.issubset(orders_df.columns):
        missing = required_cols.difference(orders_df.columns)
        raise ValueError(f"orders_df 缺少必要欄位: {missing}")

    interval = (
        orders_df[['user_id', 'days_since_prior_order']]
        .dropna(subset=['days_since_prior_order'])
        .groupby('user_id', as_index=False)
        .agg(avg_days_since_prior=('days_since_prior_order', 'mean'))
    )

    return interval

def compute_department_frequency(
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    departments_df: pd.DataFrame
) -> pd.DataFrame:
    """彙總每位用戶在各部門的購買頻率與占比。"""
    for col in ['user_id', 'product_id']:
        if col not in transactions_df.columns:
            raise ValueError(f"transactions_df 缺少必要欄位: {col}")

    if 'product_id' not in products_df.columns or 'department_id' not in products_df.columns:
        raise ValueError("products_df 需包含 'product_id' 與 'department_id' 欄位")

    if 'department_id' not in departments_df.columns:
        raise ValueError("departments_df 需包含 'department_id' 欄位")

    product_department = products_df[['product_id', 'department_id']].drop_duplicates()
    department_lookup = departments_df[['department_id', 'department']].drop_duplicates()

    transactions_with_department = (
        transactions_df.merge(product_department, on='product_id', how='left').dropna(subset=['department_id'])
    )

    department_counts = (
        transactions_with_department
        .groupby(['user_id', 'department_id'], as_index=False)
        .size()
        .rename(columns={'size': 'department_frequency'})
    )

    department_counts['department_frequency_ratio'] = (
        department_counts['department_frequency'] /
        department_counts.groupby('user_id')['department_frequency'].transform('sum')
    )

    department_counts = department_counts.merge(department_lookup, on='department_id', how='left')

    ordered_cols = [
        'user_id',
        'department_id',
        'department',
        'department_frequency',
        'department_frequency_ratio'
    ]

    return department_counts[ordered_cols]

def preprocess_data(
    orders_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    將 Instacart 格式的 orders 與 product 交易紀錄合併，並重建時間軸。

    簡述:
        1. 合併 order_products__prior 與 order_products__train。
        2. 將商品紀錄與 orders 資訊 (user_id, order_number) 合併。
        3. 處理 days_since_prior_order 的缺失值。
        4. 計算每個用戶的累積天數，並生成模擬的 'transaction_date'。

    Args:
        orders_df (pd.DataFrame): 包含 order_id, user_id, days_since_prior_order 等欄位。
        prior_df (pd.DataFrame): 歷史訂單商品明細。
        train_df (pd.DataFrame): 訓練集訂單商品明細。

    Returns:
        pd.DataFrame: 
            包含 'transaction_date', 'user_id', 'product_id', 'amount' (設為1) 的寬表，
            可直接丟入滑動視窗函式。
    """
    # 1. 合併商品明細 (Prior + Train)
    # 我們將兩者視為同一條時間軸上的連續數據
    print("正在合併 Prior 與 Train 商品明細...")
    all_order_products = pd.concat([prior_df, train_df], ignore_index=True)
    
    # 2. 關聯 User 資訊 (Merge with Orders)
    # 只保留需要的欄位以節省記憶體
    print("正在關聯 User 與 Time 資訊...")
    orders_slim = orders_df[['order_id', 'user_id', 'order_number', 'days_since_prior_order']]
    
    # Inner Join: 確保每一條商品紀錄都有對應的訂單資訊
    merged_df = pd.merge(all_order_products, orders_slim, on='order_id', how='inner')
    
    # 3. 重建時間軸 (關鍵步驟)
    print("正在重建模擬時間軸 (Pseudo-Timeline)...")
    
    # 填補第一筆訂單的 NaN 為 0
    merged_df['days_since_prior_order'] = merged_df['days_since_prior_order'].fillna(0)
    
    # 確保按照 user 和 order_number 排序
    merged_df = merged_df.sort_values(['user_id', 'order_number'])
    
    # 計算累積天數 (Cumulative Days)
    # 我們需要對 'order_id' 級別做 cumsum，而不是 product 級別
    # 先算出每個 order_id 對應的累積天數
    order_days = orders_slim.sort_values(['user_id', 'order_number']).copy()
    order_days['days_since_prior_order'] = order_days['days_since_prior_order'].fillna(0)
    
    # GroupBy User 進行累積加總
    order_days['cumulative_days'] = order_days.groupby('user_id')['days_since_prior_order'].cumsum()
    
    # 設定一個起始日期 (錨點)，例如 2020-01-01
    start_date = datetime(2020, 1, 1)
    order_days['transaction_date'] = order_days['cumulative_days'].apply(lambda x: start_date + timedelta(days=x))
    
    # 將計算好的日期 Merge 回主表
    final_df = pd.merge(merged_df, order_days[['order_id', 'transaction_date']], on='order_id', how='left')
    
    # 4. 整理最終欄位
    # Instacart 資料沒有金額，我們可以用數量 1 來代表 (用於計算 RFM 的 Monetary/Frequency)
    final_df['quantity'] = 1 
    
    selected_cols = ['transaction_date','order_id', 'user_id', 'product_id', 'quantity']
    return final_df[selected_cols]

def build_xgboost_training_data(
    train_dataset: pd.DataFrame,
    avg_purchase_interval: pd.DataFrame,
    department_frequency: pd.DataFrame,
    products_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """整合所有特徵並建立 XGBoost 訓練資料集。

    此函式將滑動視窗 RFM 特徵、用戶平均購買間隔與部門頻率特徵合併，
    產出可直接用於 XGBoost 訓練的 (X, y) 資料。

    Args:
        train_dataset (pd.DataFrame): 滑動視窗生成的樣本，需包含
            user_id, product_id, recency, frequency, monetary, label, cut_date。
        avg_purchase_interval (pd.DataFrame): 用戶平均購買間隔，需包含
            user_id, avg_days_since_prior。
        department_frequency (pd.DataFrame): 用戶部門購買頻率，需包含
            user_id, department_id, department_frequency, department_frequency_ratio。
        products_df (pd.DataFrame): 商品資訊表，需包含 product_id, department_id。

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: 特徵矩陣 (不含 label, cut_date 等非特徵欄位)。
            - y: 標籤向量 (label)。
    """
    print("--- 開始整合 XGBoost 訓練特徵 ---")

    df = train_dataset.copy()

    # 1. 合併用戶平均購買間隔
    df = df.merge(avg_purchase_interval, on='user_id', how='left')
    df['avg_days_since_prior'] = df['avg_days_since_prior'].fillna(df['avg_days_since_prior'].median())

    # 2. 合併商品所屬部門
    product_dept = products_df[['product_id', 'department_id']].drop_duplicates()
    df = df.merge(product_dept, on='product_id', how='left')

    # 3. 合併用戶在該部門的購買頻率
    dept_freq_slim = department_frequency[['user_id', 'department_id', 'department_frequency', 'department_frequency_ratio']].copy()
    df = df.merge(dept_freq_slim, on=['user_id', 'department_id'], how='left')
    df['department_frequency'] = df['department_frequency'].fillna(0)
    df['department_frequency_ratio'] = df['department_frequency_ratio'].fillna(0)

    # 4. 定義特徵欄位與標籤
    feature_cols = [
        'recency',
        'frequency',
        'monetary',
        'avg_days_since_prior',
        'department_id',
        'department_frequency',
        'department_frequency_ratio'
    ]

    # 確保所有特徵欄位存在
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要特徵欄位: {col}")

    X = df[feature_cols].copy()
    y = df['label'].copy()

    print(f"特徵矩陣形狀: {X.shape}")
    print(f"正樣本比例: {y.mean():.4%}")
    print("--- 特徵整合完成 ---")

    return X, y

def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: Dict = None
) -> Tuple[xgb.XGBClassifier, Dict]:
    """訓練 XGBoost 分類模型並回傳評估結果。

    Args:
        X (pd.DataFrame): 特徵矩陣。
        y (pd.Series): 標籤向量。
        test_size (float, optional): 測試集比例，預設 0.2。
        random_state (int, optional): 隨機種子，預設 42。
        xgb_params (Dict, optional): XGBoost 超參數，若為 None 則使用預設值。

    Returns:
        Tuple[xgb.XGBClassifier, Dict]:
            - model: 訓練完成的 XGBoost 模型。
            - metrics: 包含 accuracy, precision, recall, f1, auc 的評估指標字典。
    """

    print("--- 開始訓練 XGBoost 模型 ---")

    # 切分訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 預設超參數
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': random_state
        }

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 預測與評估
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba)
    }

    print(f"測試集評估結果:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print("--- 訓練完成 ---")

    return model, metrics, X_test, y_test, y_pred, y_proba

def save_model_and_artifacts(
    model: xgb.XGBClassifier,
    metrics: Dict,
    feature_cols: List[str],
    avg_purchase_interval: pd.DataFrame,
    department_frequency: pd.DataFrame,
    products_df: pd.DataFrame,
    output_dir: str = "./model/1/"
) -> None:
    """儲存模型與相關資料至指定目錄。

    Args:
        model (xgb.XGBClassifier): 訓練完成的模型。
        metrics (Dict): 評估指標字典。
        feature_cols (List[str]): 特徵欄位名稱列表。
        avg_purchase_interval (pd.DataFrame): 用戶平均購買間隔。
        department_frequency (pd.DataFrame): 部門購買頻率。
        products_df (pd.DataFrame): 商品資訊表。
        output_dir (str, optional): 輸出目錄，預設 "./model/1/"。
    """

    os.makedirs(output_dir, exist_ok=True)

    # 儲存模型
    model_path = os.path.join(output_dir, "xgboost_model.json")
    model.save_model(model_path)
    print(f"模型已儲存至: {model_path}")

    # 儲存評估指標
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"評估指標已儲存至: {metrics_path}")

    # 儲存特徵欄位
    feature_path = os.path.join(output_dir, "feature_cols.json")
    with open(feature_path, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    print(f"特徵欄位已儲存至: {feature_path}")

    # 儲存輔助資料 (用於推論)
    avg_interval_path = os.path.join(output_dir, "avg_purchase_interval.csv")
    avg_purchase_interval.to_csv(avg_interval_path, index=False)
    print(f"平均購買間隔已儲存至: {avg_interval_path}")

    dept_freq_path = os.path.join(output_dir, "department_frequency.csv")
    department_frequency.to_csv(dept_freq_path, index=False)
    print(f"部門購買頻率已儲存至: {dept_freq_path}")

    product_dept_path = os.path.join(output_dir, "product_department.csv")
    products_df[['product_id', 'department_id']].drop_duplicates().to_csv(product_dept_path, index=False)
    print(f"商品部門對照已儲存至: {product_dept_path}")

def plot_evaluation_charts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model: xgb.XGBClassifier,
    feature_names: List[str],
    output_dir: str = "runtime6/"
) -> None:
    """繪製模型評估圖表並儲存。

    產出圖表包含：
    1. Confusion Matrix
    2. ROC Curve
    3. Precision-Recall Curve
    4. Calibration Curve
    5. Feature Importance
    6. Cumulative Gain Chart

    Args:
        y_true (np.ndarray): 真實標籤。
        y_pred (np.ndarray): 預測標籤。
        y_proba (np.ndarray): 預測機率。
        model (xgb.XGBClassifier): 訓練完成的模型。
        feature_names (List[str]): 特徵名稱列表。
        output_dir (str, optional): 輸出目錄，預設 "./model/1/"。
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Repurchase', 'Repurchase'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("已儲存: confusion_matrix.png")

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)
    print("已儲存: roc_curve.png")

    # 3. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.4f})')
    ax.axhline(y=y_true.mean(), color='red', linestyle='--', lw=1, label=f'Baseline (Positive Rate = {y_true.mean():.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)
    print("已儲存: precision_recall_curve.png")

    # 4. Calibration Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', lw=2, color='blue', label='XGBoost')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150)
    plt.close(fig)
    print("已儲存: calibration_curve.png")

    # 5. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color='steelblue')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("已儲存: feature_importance.png")

    # 6. Cumulative Gain Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_indices = np.argsort(y_proba)[::-1]
    sorted_y_true = y_true.values[sorted_indices] if hasattr(y_true, 'values') else y_true[sorted_indices]
    cumulative_gains = np.cumsum(sorted_y_true) / sorted_y_true.sum()
    percentiles = np.arange(1, len(cumulative_gains) + 1) / len(cumulative_gains)
    ax.plot(percentiles, cumulative_gains, lw=2, color='purple', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    ax.set_xlabel('Proportion of Sample', fontsize=12)
    ax.set_ylabel('Cumulative Gain', fontsize=12)
    ax.set_title('Cumulative Gain Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_gain_chart.png"), dpi=150)
    plt.close(fig)
    print("已儲存: cumulative_gain_chart.png")

    print(f"所有評估圖表已儲存至: {output_dir}")

class RepurchasePredictor:
    """回購機率預測器，載入訓練好的模型進行推論。

    Example:
        >>> predictor = RepurchasePredictor("./model/1/")
        >>> proba = predictor.predict(user_id=123, product_id=456)
        >>> print(f"回購機率: {proba:.4f}")
    """

    def __init__(self, model_dir: str = "./model/1/"):
        """初始化預測器，載入模型與輔助資料。

        Args:
            model_dir (str): 模型目錄路徑。
        """
        print("--- 載入回購機率預測器 ---")

        self.model_dir = model_dir

        # 載入模型
        model_path = os.path.join(model_dir, "xgboost_model.json")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # 載入特徵欄位
        feature_path = os.path.join(model_dir, "feature_cols.json")
        with open(feature_path, 'r', encoding='utf-8') as f:
            self.feature_cols = json.load(f)

        # 載入輔助資料
        self.avg_purchase_interval = pd.read_csv(
            os.path.join(model_dir, "avg_purchase_interval.csv")
        )
        self.department_frequency = pd.read_csv(
            os.path.join(model_dir, "department_frequency.csv")
        )
        self.product_department = pd.read_csv(
            os.path.join(model_dir, "product_department.csv")
        )

        # 建立查詢索引
        self._build_lookup_indices()

        print(f"預測器已載入，模型路徑: {model_dir}")

    def _build_lookup_indices(self):
        """建立快速查詢索引。"""
        self.avg_interval_dict = self.avg_purchase_interval.set_index('user_id')['avg_days_since_prior'].to_dict()
        self.product_dept_dict = self.product_department.set_index('product_id')['department_id'].to_dict()

        # 部門頻率需要 (user_id, department_id) 的複合索引
        self.dept_freq_dict = {}
        for _, row in self.department_frequency.iterrows():
            key = (row['user_id'], row['department_id'])
            self.dept_freq_dict[key] = {
                'department_frequency': row['department_frequency'],
                'department_frequency_ratio': row['department_frequency_ratio']
            }

        # 計算全域中位數用於填補缺失值
        self.median_avg_interval = self.avg_purchase_interval['avg_days_since_prior'].median()

    def predict(
        self,
        user_id: int,
        product_id: int,
        recency: int = 0,
        frequency: int = 1,
        monetary: float = 1.0
    ) -> float:
        """預測指定用戶對指定商品的回購機率。

        Args:
            user_id (int): 用戶 ID。
            product_id (int): 商品 ID。
            recency (int, optional): 距離上次購買天數，預設 0。
            frequency (int, optional): 購買頻次，預設 1。
            monetary (float, optional): 購買金額，預設 1.0。

        Returns:
            float: 回購機率 (0~1)。
        """
        print("--- 開始單筆預測 ---")
        print(f"用戶 ID: {user_id}, 商品 ID: {product_id}")
        # 取得用戶平均購買間隔
        avg_days = self.avg_interval_dict.get(user_id, self.median_avg_interval)

        # 取得商品所屬部門
        dept_id = self.product_dept_dict.get(product_id, 0)

        # 取得用戶在該部門的購買頻率
        dept_key = (user_id, dept_id)
        dept_info = self.dept_freq_dict.get(dept_key, {'department_frequency': 0, 'department_frequency_ratio': 0})

        # 組裝特徵
        features = {
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'avg_days_since_prior': avg_days,
            'department_id': dept_id,
            'department_frequency': dept_info['department_frequency'],
            'department_frequency_ratio': dept_info['department_frequency_ratio']
        }

        X = pd.DataFrame([features])[self.feature_cols]
        proba = self.model.predict_proba(X)[0, 1]

        return float(proba)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """批量預測回購機率。

        Args:
            df (pd.DataFrame): 包含 user_id, product_id, recency, frequency, monetary 的 DataFrame。

        Returns:
            np.ndarray: 回購機率陣列。
        """
        result = df.copy()

        # 合併平均購買間隔
        result = result.merge(self.avg_purchase_interval, on='user_id', how='left')
        result['avg_days_since_prior'] = result['avg_days_since_prior'].fillna(self.median_avg_interval)

        # 合併商品部門
        result = result.merge(self.product_department, on='product_id', how='left')
        result['department_id'] = result['department_id'].fillna(0)

        # 合併部門頻率
        dept_freq_slim = self.department_frequency[['user_id', 'department_id', 'department_frequency', 'department_frequency_ratio']]
        result = result.merge(dept_freq_slim, on=['user_id', 'department_id'], how='left')
        result['department_frequency'] = result['department_frequency'].fillna(0)
        result['department_frequency_ratio'] = result['department_frequency_ratio'].fillna(0)

        X = result[self.feature_cols]
        probas = self.model.predict_proba(X)[:, 1]

        return probas

if __name__ == "__main__":
    DATA_PATH = "./data/"
    AISLES = pd.read_csv(DATA_PATH + "aisles.csv")
    DEPARTMENTS = pd.read_csv(DATA_PATH + "departments.csv")
    PRODUCTS = pd.read_csv(DATA_PATH + "products.csv")
    ORDERS = pd.read_csv(DATA_PATH + "orders.csv")
    ORDER_PRODUCTS_PRIOR = pd.read_csv(DATA_PATH + "order_products__prior.csv")
    ORDER_PRODUCTS_TRAIN = pd.read_csv(DATA_PATH + "order_products__train.csv")
    CLEAN_TRANSACTIONS = pd.read_csv(DATA_PATH + "clean_transactions1.csv", parse_dates=['transaction_date'])
    print("Data loaded successfully.")
    print(len(ORDER_PRODUCTS_PRIOR), "prior order products loaded.")
    print(len(ORDER_PRODUCTS_TRAIN), "train order products loaded.")
    print(CLEAN_TRANSACTIONS.head())
    print(f"Min date: {CLEAN_TRANSACTIONS['transaction_date'].min()}")
    print(f"Max date: {CLEAN_TRANSACTIONS['transaction_date'].max()}")

    avg_purchase_interval = compute_average_purchase_interval(ORDERS)
    print("\n--- 平均購買間隔 (前 5 筆) ---")
    print(avg_purchase_interval.head())

    department_frequency = compute_department_frequency(
        CLEAN_TRANSACTIONS,
        PRODUCTS,
        DEPARTMENTS
    )
    print("\n--- 部門購買頻率 (前 5 筆) ---")
    print(department_frequency.head())

    train_dataset = build_sliding_window_rfm(
        CLEAN_TRANSACTIONS,
        date_col='transaction_date',
        user_col='user_id',
        item_col='product_id',
        amount_col='quantity',
        observation_days=60,  # 觀察過去 60 天
        label_days=30,        # 預測未來 30 天
        stride_days=15        # 每 15 天滑動一次
    )

    # 建立 XGBoost 訓練資料
    X, y = build_xgboost_training_data(
        train_dataset,
        avg_purchase_interval,
        department_frequency,
        PRODUCTS
    )

    # 訓練模型
    model, metrics, X_test, y_test, y_pred, y_proba = train_xgboost_model(X, y)

    # 儲存模型與輔助資料
    MODEL_DIR = "./model/1/"
    save_model_and_artifacts(
        model=model,
        metrics=metrics,
        feature_cols=X.columns.tolist(),
        avg_purchase_interval=avg_purchase_interval,
        department_frequency=department_frequency,
        products_df=PRODUCTS,
        output_dir=MODEL_DIR
    )

    # 繪製評估圖表
    plot_evaluation_charts(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model=model,
        feature_names=X.columns.tolist(),
        output_dir=MODEL_DIR
    )

    # 顯示特徵重要性
    print("\n--- 特徵重要性 ---")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance)

    # 測試預測器
    print("\n--- 測試 RepurchasePredictor ---")
    predictor = RepurchasePredictor(MODEL_DIR)
    sample_user = train_dataset['user_id'].iloc[0]
    sample_product = train_dataset['product_id'].iloc[0]
    proba = predictor.predict(
        user_id=sample_user,
        product_id=sample_product,
        recency=10,
        frequency=3,
        monetary=3.0
    )
    print(f"用戶 {sample_user} 對商品 {sample_product} 的回購機率: {proba:.4f}")

    print("\n--- 最終 XGBoost 訓練資料 ---")
    print(train_dataset.head())
    print(f"特徵欄位: {train_dataset.columns.tolist()}")

