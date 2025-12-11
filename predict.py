import pandas as pd
from xgboost import RepurchasePredictor
from mba import recommend_products
import random

MBA_PATH = "./model/mba/"
XGBOOST_PATH = "./model/Xgboost/"

def mba_recommender(product: int | str):
    recommendations = recommend_products(
        input_list=[product],
        model_path=MBA_PATH,
        top_n=3
    )
    return recommendations

if __name__ == "__main__":
    predictor = RepurchasePredictor(XGBOOST_PATH)
    while True:
        product = input("請輸入商品名稱或ID進行推薦: ")
        try:
            product = int(product)
        except ValueError:
            pass
        if isinstance(product, str) and product.strip() == "":
            product = random.randint(1, 49688)
        proba = predictor.predict(
            user_id=558,        # 用戶 ID
            product_id=product,    # 商品 ID
            recency=10,         # 距離上次購買天數
            frequency=3,        # 購買頻次
            monetary=3.0        # 購買金額
        )
        print(f"用戶 558 對商品 {product} 的回購機率: {proba:.4f} ({proba*100:.1f}%)")
        print(f"購買 {product} 的人也買了:")
        recommendations = mba_recommender(product)
        for i, name in enumerate(recommendations, 1):
            print(f"  {i}. {name}")