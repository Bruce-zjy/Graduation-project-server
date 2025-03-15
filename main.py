from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from flask_cors import CORS  # 🔥 添加 CORS 解决前端跨域问题

app = Flask(__name__)
CORS(app, supports_credentials=True, origins="http://localhost:3000")

# 📌 1️⃣ 读取 `preprocessor.pkl` 和 `models`
MODEL_PATH = "models"
PREPROCESSOR_PATH = os.path.join(MODEL_PATH, "preprocessor.pkl")

if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    expected_feature_count = len(preprocessor.get_feature_names_out())  # 获取转换后的特征数
    print(f"✅ `preprocessor.pkl` 加载成功！转换后特征维度: {expected_feature_count}")
else:
    raise FileNotFoundError("❌ `preprocessor.pkl` 未找到！请检查文件路径")

# 📌 2️⃣ 加载所有模型
models = {
    file.replace(".pkl", ""): joblib.load(os.path.join(MODEL_PATH, file))
    for file in os.listdir(MODEL_PATH) if file.endswith(".pkl")
}
print(f"✅ 加载模型完成: {list(models.keys())}")

# 📌 3️⃣ 定义特征顺序
num_features = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "agent", "company",
    "required_car_parking_spaces", "total_of_special_requests", "adr"
]

cat_features = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"
]

column_names = num_features + cat_features
original_feature_count = len(column_names)

@app.route("/")
def home():
    return "🚀 Flask API 运行中！"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        model_names = data.get("models", [])
        features = data.get("features", [])

        if not isinstance(model_names, list):
            return jsonify({"error": "❌ 模型列表应为数组格式"}), 400

        if not features:
            return jsonify({"error": "❌ 输入特征为空！请提供正确的特征数据"}), 400

        if len(features) != len(column_names):
            return jsonify({
                "error": f"❌ 输入特征数量错误！期望 {len(column_names)} 个特征，你提供了 {len(features)} 个",
                "received_features": features,
                "expected_columns": column_names
            }), 400

        print(f"🔥 输入数据 (原始): {features}")

        # ✅ 转换为 Pandas DataFrame
        df_features = pd.DataFrame([features], columns=column_names, dtype=object)

        # ✅ 处理数值类型
        for col in num_features:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

        df_features[cat_features] = df_features[cat_features].astype(str)

        print(f"✅ `df_features` 数据类型:\n{df_features.dtypes}")

        # **🔥 预处理 `features`**
        if preprocessor:
            transformed_features = preprocessor.transform(df_features)
            transformed_features = np.asarray(transformed_features, dtype=np.float32)
            print(f"🔥 预处理后特征维度: {transformed_features.shape}")
        else:
            return jsonify({"error": "❌ 预处理器未找到，请确保 `preprocessor.pkl` 存在！"}), 400

        if transformed_features.shape[1] != expected_feature_count:
            return jsonify({
                "error": f"❌ 预处理后的特征维度 ({transformed_features.shape[1]}) 与模型期望 ({expected_feature_count}) 不匹配！"
            }), 400

        predictions = {}
        for model_name in model_names:
            if model_name not in models:
                predictions[model_name] = "❌ 该模型不存在"
            else:
                model = models[model_name]

                try:
                    print(f"🚀 运行模型: {model_name}")

                    feature_names = preprocessor.get_feature_names_out()
                    df_transformed = pd.DataFrame(transformed_features, columns=[
                        col.replace("num__", "").replace("cat__", "").replace("_Unknown", "")
                        for col in feature_names
                    ])

                    print(f"🔥 修正后的 `df_transformed` 列名:\n{df_transformed.columns}")

                    # ✅ **支持分类模型 `predict_proba`**
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(df_transformed)[0]
                        final_prediction = 1 if proba[1] > 0.5 else 0
                        predictions[model_name] = {
                            "prediction": final_prediction,
                            "probabilities": {
                                "0": round(proba[0], 4),
                                "1": round(proba[1], 4)
                            }
                        }
                    else:
                        predictions[model_name] = int(model.predict(df_transformed)[0])
                except Exception as e:
                    predictions[model_name] = f"❌ 预测失败: {str(e)}"
                    print(f"⚠️ `{model_name}` 预测时发生错误: {e}")

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
