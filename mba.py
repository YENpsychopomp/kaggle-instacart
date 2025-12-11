"""
Instacart MBA Recommendation System V1
=====================================
商品組合分析與推薦系統

功能：
1. train_mba_model(): 訓練關聯規則模型 (使用 PySpark FPGrowth 加速)
2. recommend_products(): 基於購物籃推薦商品
"""

import os
import json
import pickle
import warnings
from typing import List, Union, Dict, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================
# 全域配置
# ============================================================
DATA_PATH = "./data/"
MODEL_PATH = "./model/mba/"

# 預設參數
DEFAULT_PARAMS = {
    'mba_algorithm': 'FPGrowth',
    'min_support': 0.0005,
    'min_confidence': 0.2,
    'max_k': 2,
    'recommend_top_n': 3
}


# ============================================================
# Phase 1: 資料前處理與整合
# ============================================================
def load_product_mappings(data_path: str = DATA_PATH) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    載入商品映射字典（不需要 Spark，用 Pandas 即可）。
    
    Args:
        data_path: 資料目錄路徑
        
    Returns:
        Tuple[Dict, Dict, pd.DataFrame]:
            - id_to_name: product_id -> product_name 映射
            - name_to_id: product_name -> product_id 映射
            - products_full: 完整商品資料表
    """
    print("\n[1.1] 載入商品對照表...")
    products = pd.read_csv(os.path.join(data_path, "products.csv"))
    departments = pd.read_csv(os.path.join(data_path, "departments.csv"))
    aisles = pd.read_csv(os.path.join(data_path, "aisles.csv"))
    
    products_full = products.merge(departments, on='department_id', how='left')
    products_full = products_full.merge(aisles, on='aisle_id', how='left')
    
    id_to_name = dict(zip(products_full['product_id'], products_full['product_name']))
    name_to_id = dict(zip(products_full['product_name'], products_full['product_id']))
    
    print(f"   ✓ 商品數量: {len(id_to_name):,}")
    
    return id_to_name, name_to_id, products_full


# ============================================================
# Phase 2: 使用 PySpark FPGrowth 訓練模型
# ============================================================
def train_mba_model(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    min_support: float = 0.001,
    min_confidence: float = 0.1,
    limit_rows: int = None,  # None = 使用全部資料
    spark_driver_memory: str = "8g",
    spark_executor_memory: str = "8g"
) -> pd.DataFrame:
    """
    使用 PySpark FPGrowth 訓練關聯規則模型（可處理全部 3300 萬筆交易）。
    
    Args:
        data_path: 資料目錄路徑
        model_path: 模型儲存路徑
        min_support: 最小支持度閾值
        min_confidence: 最小信賴度閾值
        limit_rows: 限制資料筆數 (None 表示全部)
        spark_driver_memory: Spark Driver 記憶體
        spark_executor_memory: Spark Executor 記憶體
        
    Returns:
        pd.DataFrame: 關聯規則 DataFrame
    """
    import time
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import collect_list, col, size
    from pyspark.ml.fpm import FPGrowth
    
    print("\n" + "=" * 60)
    print("🔧 Phase 2: 模型訓練與規則生成 (PySpark FPGrowth)")
    print("=" * 60)
    print(f"\n參數設定:")
    print(f"   min_support: {min_support}")
    print(f"   min_confidence: {min_confidence}")
    print(f"   limit_rows: {'全部資料' if limit_rows is None else f'{limit_rows:,}'}")
    print(f"   spark_driver_memory: {spark_driver_memory}")
    
    # 1. 載入商品映射（使用 Pandas）
    id_to_name, name_to_id, products_full = load_product_mappings(data_path)
    
    # 2. 初始化 Spark Session
    print("\n[2.1] 初始化 Spark Session...")
    
    # Windows 兼容性設置
    import os as os_module
    os_module.environ['PYSPARK_PYTHON'] = 'python'
    os_module.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
    
    spark = SparkSession.builder \
        .appName("MBA_FPGrowth_Full") \
        .master("local[*]") \
        .config("spark.driver.memory", spark_driver_memory) \
        .config("spark.executor.memory", spark_executor_memory) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("   ✓ Spark Session 已初始化")
    
    try:
        # 3. 載入交易資料
        print("\n[2.2] 載入交易資料...")
        df_prior = spark.read.csv(
            os.path.join(data_path, "order_products__prior.csv"),
            header=True, inferSchema=True
        )
        df_train = spark.read.csv(
            os.path.join(data_path, "order_products__train.csv"),
            header=True, inferSchema=True
        )
        
        # 合併資料
        df_total = df_prior.union(df_train)
        
        # 計算總筆數
        if limit_rows:
            print(f"   ⚠️ 限制資料為 {limit_rows:,} 筆")
            df_total = df_total.limit(limit_rows)
        
        total_count = df_total.count()
        print(f"   ✓ 總交易筆數: {total_count:,}")
        
        # 4. 建立購物籃（以 order_id 分組）
        print("\n[2.3] 建立購物籃資料...")
        basket_data = df_total.select("order_id", "product_id") \
            .groupBy("order_id") \
            .agg(collect_list("product_id").alias("items"))
        
        # 過濾只有 1 個商品的訂單
        basket_data = basket_data.filter(size(col("items")) >= 2)
        basket_count = basket_data.count()
        print(f"   ✓ 有效購物籃數: {basket_count:,}")
        
        # 5. 訓練 FPGrowth 模型
        print(f"\n[2.4] 訓練 FPGrowth 模型...")
        start_time = time.time()
        
        fpGrowth = FPGrowth(
            itemsCol="items",
            minSupport=min_support,
            minConfidence=min_confidence
        )
        model = fpGrowth.fit(basket_data)
        
        training_time = time.time() - start_time
        print(f"   ✓ 訓練完成！耗時: {training_time:.2f} 秒")
        
        # 6. 取得頻繁項集與關聯規則
        print("\n[2.5] 提取關聯規則...")
        
        # 直接取得關聯規則（避免計算全部頻繁項集以節省記憶體）
        rules_spark = model.associationRules
        
        # 先取前 N 條規則以避免記憶體問題
        print("   → 按 Lift 排序並取得規則...")
        rules_spark = rules_spark.sort(col("lift").desc())
        
        # 限制規則數量（最多 10000 條）
        rules_spark = rules_spark.limit(10000)
        
        # 7. 轉換為 Pandas DataFrame
        print("\n[2.6] 轉換規則格式...")
        rules_pd = rules_spark.toPandas()
        
        rules_count = len(rules_pd)
        print(f"   ✓ 關聯規則數量: {rules_count:,}")
        
        if rules_count == 0:
            print("   ⚠️ 未生成任何規則，請嘗試降低 min_support 或 min_confidence")
            return pd.DataFrame()
        
        # 將 antecedent 和 consequent 轉為 frozenset（與 mlxtend 格式相容）
        rules_pd['antecedents'] = rules_pd['antecedent'].apply(lambda x: frozenset(x))
        rules_pd['consequents'] = rules_pd['consequent'].apply(lambda x: frozenset(x))
        
        # 重新命名欄位以保持相容性
        rules_pd = rules_pd.rename(columns={'support': 'support'})
        
        # 按 Lift 排序
        rules_pd = rules_pd.sort_values('lift', ascending=False).reset_index(drop=True)
        
        print(f"\n   規則統計:")
        print(f"   → Support 範圍: [{rules_pd['support'].min():.6f}, {rules_pd['support'].max():.6f}]")
        print(f"   → Confidence 範圍: [{rules_pd['confidence'].min():.4f}, {rules_pd['confidence'].max():.4f}]")
        print(f"   → Lift 範圍: [{rules_pd['lift'].min():.2f}, {rules_pd['lift'].max():.2f}]")
        
        # 8. 儲存模型
        print(f"\n[2.7] 儲存模型...")
        os.makedirs(model_path, exist_ok=True)
        
        # 儲存關聯規則
        rules_path = os.path.join(model_path, "mba_rules_model.pkl")
        with open(rules_path, 'wb') as f:
            pickle.dump(rules_pd, f)
        print(f"   ✓ 規則已儲存: {rules_path}")
        
        # 儲存商品映射字典
        mappings = {
            'id_to_name': id_to_name,
            'name_to_id': name_to_id
        }
        mappings_path = os.path.join(model_path, "product_mappings.pkl")
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
        print(f"   ✓ 商品映射已儲存: {mappings_path}")
        
        # 儲存參數設定
        params = {
            'min_support': min_support,
            'min_confidence': min_confidence,
            'limit_rows': limit_rows,
            'total_transactions': total_count,
            'basket_count': basket_count,
            'n_rules': len(rules_pd),
            'n_products': len(id_to_name),
            'training_time_seconds': training_time
        }
        params_path = os.path.join(model_path, "training_params.json")
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"   ✓ 訓練參數已儲存: {params_path}")
        
        print("\n" + "=" * 60)
        print("✅ 模型訓練完成！")
        print("=" * 60)
        
        # 顯示前 10 條規則
        print("\n📋 前 10 條規則 (按 Lift 排序):")
        print("-" * 80)
        for idx, row in rules_pd.head(10).iterrows():
            ant_names = [id_to_name.get(pid, str(pid)) for pid in row['antecedents']]
            con_names = [id_to_name.get(pid, str(pid)) for pid in row['consequents']]
            print(f"{idx+1:2}. {ant_names} → {con_names}")
            print(f"    Support: {row['support']:.6f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.2f}")
        
        return rules_pd
        
    finally:
        # 確保 Spark Session 被關閉
        spark.stop()
        print("\n   ✓ Spark Session 已關閉")


# ============================================================
# Phase 3: 推薦查詢函式
# ============================================================
def recommend_products(
    input_list: List[Union[int, str]],
    model_path: str = MODEL_PATH,
    top_n: int = DEFAULT_PARAMS['recommend_top_n']
) -> List[str]:
    """
    基於購物籃推薦商品。
    
    可接受商品 ID (int) 或商品名稱 (str) 的混合輸入。
    
    Args:
        input_list: 購物籃商品列表 (可混合 ID 和名稱)
        model_path: 模型目錄路徑
        top_n: 推薦商品數量
        
    Returns:
        List[str]: 推薦的商品名稱列表
    """
    # 1. 載入模型和映射
    rules_path = os.path.join(model_path, "mba_rules_model.pkl")
    mappings_path = os.path.join(model_path, "product_mappings.pkl")
    
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"找不到模型檔案: {rules_path}，請先執行 train_mba_model()")
    
    with open(rules_path, 'rb') as f:
        rules = pickle.load(f)
    
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    id_to_name = mappings['id_to_name']
    name_to_id = mappings['name_to_id']
    
    # 2. 解析輸入，統一轉換為 product_id 集合
    input_ids: Set[int] = set()
    
    for item in input_list:
        if isinstance(item, int):
            # 輸入是 product_id
            if item in id_to_name:
                input_ids.add(item)
            else:
                print(f"   ⚠️ 未知的商品 ID: {item}")
        elif isinstance(item, str):
            # 輸入是 product_name
            if item in name_to_id:
                input_ids.add(name_to_id[item])
            else:
                # 嘗試模糊匹配
                matches = [name for name in name_to_id.keys() if item.lower() in name.lower()]
                if matches:
                    matched_name = matches[0]
                    input_ids.add(name_to_id[matched_name])
                    print(f"   ℹ️ 模糊匹配: '{item}' → '{matched_name}'")
                else:
                    print(f"   ⚠️ 未知的商品名稱: {item}")
    
    if not input_ids:
        print("   ❌ 無法識別任何輸入商品")
        return []
    
    # 3. 篩選規則
    # 條件 1: antecedent 必須是 input_ids 的子集
    # 條件 2: consequent 不能包含 input_ids 中的任何商品
    
    candidate_products: Dict[int, float] = {}  # product_id -> max_lift
    
    for _, row in rules.iterrows():
        antecedent = set(row['antecedents'])
        consequent = set(row['consequents'])
        
        # 檢查條件 1: antecedent ⊆ input_ids
        if not antecedent.issubset(input_ids):
            continue
        
        # 檢查條件 2: consequent ∩ input_ids = ∅
        if consequent.intersection(input_ids):
            continue
        
        # 符合條件，記錄推薦商品與其 Lift
        lift = row['lift']
        for product_id in consequent:
            if product_id not in candidate_products or candidate_products[product_id] < lift:
                candidate_products[product_id] = lift
    
    # 4. 按 Lift 排序並取 top_n
    sorted_products = sorted(candidate_products.items(), key=lambda x: x[1], reverse=True)
    top_products = sorted_products[:top_n]
    
    # 5. 轉換為商品名稱
    recommendations = [id_to_name.get(pid, f"Unknown({pid})") for pid, _ in top_products]
    
    return recommendations


def recommend_products_verbose(
    input_list: List[Union[int, str]],
    model_path: str = MODEL_PATH,
    top_n: int = DEFAULT_PARAMS['recommend_top_n']
) -> Dict:
    """
    推薦商品（詳細版本，包含完整資訊）。
    
    Args:
        input_list: 購物籃商品列表
        model_path: 模型目錄路徑
        top_n: 推薦商品數量
        
    Returns:
        Dict: 包含推薦結果和詳細資訊的字典
    """
    # 載入模型和映射
    rules_path = os.path.join(model_path, "mba_rules_model.pkl")
    mappings_path = os.path.join(model_path, "product_mappings.pkl")
    
    with open(rules_path, 'rb') as f:
        rules = pickle.load(f)
    
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    id_to_name = mappings['id_to_name']
    name_to_id = mappings['name_to_id']
    
    # 解析輸入
    input_ids: Set[int] = set()
    input_parsed = []
    
    for item in input_list:
        if isinstance(item, int):
            if item in id_to_name:
                input_ids.add(item)
                input_parsed.append({
                    'original': item,
                    'product_id': item,
                    'product_name': id_to_name[item]
                })
        elif isinstance(item, str):
            if item in name_to_id:
                pid = name_to_id[item]
                input_ids.add(pid)
                input_parsed.append({
                    'original': item,
                    'product_id': pid,
                    'product_name': item
                })
            else:
                matches = [name for name in name_to_id.keys() if item.lower() in name.lower()]
                if matches:
                    matched_name = matches[0]
                    pid = name_to_id[matched_name]
                    input_ids.add(pid)
                    input_parsed.append({
                        'original': item,
                        'product_id': pid,
                        'product_name': matched_name,
                        'fuzzy_match': True
                    })
    
    # 篩選規則並收集詳細資訊
    candidate_products: Dict[int, Dict] = {}
    matched_rules = []
    
    for _, row in rules.iterrows():
        antecedent = set(row['antecedents'])
        consequent = set(row['consequents'])
        
        if not antecedent.issubset(input_ids):
            continue
        if consequent.intersection(input_ids):
            continue
        
        matched_rules.append({
            'antecedent': [id_to_name.get(pid, str(pid)) for pid in antecedent],
            'consequent': [id_to_name.get(pid, str(pid)) for pid in consequent],
            'support': float(row['support']),
            'confidence': float(row['confidence']),
            'lift': float(row['lift'])
        })
        
        lift = row['lift']
        confidence = row['confidence']
        for product_id in consequent:
            if product_id not in candidate_products or candidate_products[product_id]['lift'] < lift:
                candidate_products[product_id] = {
                    'product_id': int(product_id),
                    'product_name': id_to_name.get(product_id, f"Unknown({product_id})"),
                    'lift': float(lift),
                    'confidence': float(confidence)
                }
    
    # 排序並取 top_n
    sorted_products = sorted(candidate_products.values(), key=lambda x: x['lift'], reverse=True)
    recommendations = sorted_products[:top_n]
    
    return {
        'input': input_parsed,
        'input_product_ids': list(input_ids),
        'matched_rules_count': len(matched_rules),
        'recommendations': recommendations,
        'top_matched_rules': matched_rules[:5]  # 只顯示前 5 條匹配規則
    }


# ============================================================
# 測試函式
# ============================================================
def run_test_cases(model_path: str = MODEL_PATH) -> Dict:
    """
    執行測試案例，驗證推薦函式。
    
    Returns:
        Dict: 測試結果
    """
    print("\n" + "=" * 60)
    print("🧪 Phase 3: 推薦查詢測試")
    print("=" * 60)
    
    # 測試案例：混合 ID 和名稱
    test_cases = [
        {
            "name": "測試案例 1: 純商品 ID (香蕉相關)",
            "input": [24852],  # Banana
            "top_n": 5
        },
        {
            "name": "測試案例 2: 純商品名稱 (有機蔬果)",
            "input": ["Organic Cilantro"],  # 香菜
            "top_n": 3
        },
        {
            "name": "測試案例 3: 混合輸入 (ID + 名稱)",
            "input": ["Organic Raspberries", 21137],  # 覆盆子 + Organic Strawberries
            "top_n": 5
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\n📌 {case['name']}")
        print(f"   輸入: {case['input']}")
        
        try:
            # 使用詳細版本
            result = recommend_products_verbose(
                input_list=case['input'],
                model_path=model_path,
                top_n=case['top_n']
            )
            
            print(f"   解析後商品:")
            for item in result['input']:
                print(f"      → {item['original']} → {item['product_name']} (ID: {item['product_id']})")
            
            print(f"   匹配規則數: {result['matched_rules_count']}")
            print(f"   推薦結果:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"      {i}. {rec['product_name']} (Lift: {rec['lift']:.2f})")
            
            results[case['name']] = {
                'input': case['input'],
                'parsed_input': result['input'],
                'recommendations': [r['product_name'] for r in result['recommendations']],
                'matched_rules_count': result['matched_rules_count']
            }
            
        except Exception as e:
            print(f"   ❌ 錯誤: {e}")
            results[case['name']] = {'error': str(e)}
    
    # 儲存測試結果
    output_path = os.path.join(model_path, "recommendation_test_output.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 測試結果已儲存: {output_path}")
    
    return results


# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🛒 Instacart MBA Recommendation System V1 (PySpark)")
    print("=" * 60)
    
    # Phase 1 & 2: 使用 PySpark FPGrowth 訓練模型（全部資料）
    rules = train_mba_model(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        min_support=0.005,       # 0.5% 支持度（約 16000 筆訂單）
        min_confidence=0.1,      # 10% 信賴度
        limit_rows=None,         # None = 使用全部 3300 萬筆資料
        spark_driver_memory="12g",
        spark_executor_memory="12g"
    )
    
    # Phase 3: 執行測試
    if len(rules) > 0:
        test_results = run_test_cases(MODEL_PATH)
        
        print("\n" + "=" * 60)
        print("🎉 所有任務完成！")
        print("=" * 60)
        print(f"\n📁 輸出檔案:")
        print(f"   → {MODEL_PATH}mba_rules_model.pkl")
        print(f"   → {MODEL_PATH}product_mappings.pkl")
        print(f"   → {MODEL_PATH}training_params.json")
        print(f"   → {MODEL_PATH}recommendation_test_output.json")
