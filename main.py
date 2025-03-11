from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# 📌 1️⃣ 读取 `preprocessor.pkl` 和 `models`
MODEL_PATH = "models"
PREPROCESSOR_PATH = os.path.join(MODEL_PATH, "preprocessor.pkl")

# ✅ **检查 `preprocessor.pkl` 是否存在**
if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    expected_feature_count = len(preprocessor.get_feature_names_out())  # 获取转换后的特征数
    print(f"✅ `preprocessor.pkl` 加载成功！转换后特征维度: {expected_feature_count}")
else:
    raise FileNotFoundError("❌ `preprocessor.pkl` 未找到！请检查文件路径")

# 📌 **2️⃣ 加载模型**
models = {
    file.replace(".pkl", ""): joblib.load(os.path.join(MODEL_PATH, file))
    for file in os.listdir(MODEL_PATH) if file.endswith(".pkl")
}
print(f"✅ 加载模型完成: {list(models.keys())}")

# 📌 **3️⃣ 设定正确的 `column_names`（确保顺序和训练时一致）**
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

column_names = num_features + cat_features  # **输入特征的列名**
original_feature_count = len(column_names)

# 🚨 **检查 `models` 是否包含 `preprocessor`**
model_filename = os.path.join(MODEL_PATH, "RF_model.pkl")
model = joblib.load(model_filename)

if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
    print(f"✅ `RF_model.pkl` 训练时已包含 `preprocessor.pkl`！")
else:
    print(f"⚠️ `RF_model.pkl` **未包含 `preprocessor.pkl`**，可能导致特征不匹配！")


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
            return jsonify({"error": "模型列表应为数组格式"}), 400

        # ✅ **检查 `features` 是否为空**
        if not features:
            return jsonify({"error": "输入特征为空！请提供正确的特征数据"}), 400

        # ✅ **检查 `features` 长度**
        if len(features) != len(column_names):
            return jsonify({
                "error": f"输入特征数量错误！期望 {len(column_names)} 个特征，你提供了 {len(features)} 个",
                "received_features": features,  # 🔥 记录传入的特征数据，方便调试
                "expected_columns": column_names  # 🔥 记录期望的列
            }), 400

        print(f"🔥 输入数据 (原始): {features}")

        # ✅ **显式转换 `features` 为 DataFrame**
        df_features = pd.DataFrame([features], columns=column_names, dtype=object)

        # ✅ **转换数据类型**
        for col in num_features:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')  # 强制转换为 float
        df_features[cat_features] = df_features[cat_features].astype(str)  # 确保类别列是 `str`

        print(f"✅ `df_features` 修正后数据类型:\n{df_features.dtypes}")

        # ✅ **检查 `df_features` 是否仍是 Pandas DataFrame**
        print(f"🔥 `df_features` 类型: {type(df_features)}")

        if not isinstance(df_features, pd.DataFrame):
            df_features = pd.DataFrame(df_features, columns=column_names)
            print("✅ `df_features` 已转换回 Pandas DataFrame")

        # **🔥 预处理 `features`**
        if preprocessor:
            transformed_features = preprocessor.transform(df_features)

            # 🚨 **确保 `transformed_features` 是 `numpy.ndarray`**
            transformed_features = np.asarray(transformed_features, dtype=np.float32)
            print(f"🔥 预处理后特征维度: {transformed_features.shape}")
            print(f"✅ `transformed_features` 类型: {type(transformed_features)}")
        else:
            return jsonify({"error": "预处理器未找到，请确保 `preprocessor.pkl` 存在！"}), 400

        # 🚨 **确保转换后的维度正确**
        if transformed_features.shape[1] != expected_feature_count:
            return jsonify({
                "error": f"转换后的特征维度 ({transformed_features.shape[1]}) 与模型期望 ({expected_feature_count}) 不匹配！"
            }), 400

        predictions = {}
        for model_name in model_names:
            if model_name not in models:
                predictions[model_name] = "模型不存在"
            else:
                model = models[model_name]

                try:
                    print(f"🚀 预测模型: {model_name}")
                    print(
                        f"🔍 `model.predict()` 输入维度: {transformed_features.shape}, 类型: {type(transformed_features)}")

                    # 🚀 **修正 `df_transformed`，让列名与 `RF_model` 训练时一致**
                    feature_names = preprocessor.get_feature_names_out()
                    df_transformed = pd.DataFrame(transformed_features, columns=[
                        col.replace("num__", "").replace("cat__", "").replace("_Unknown", "")
                        for col in feature_names
                    ])

                    print(f"🔥 修正后的 `df_transformed` 列名:\n{df_transformed.columns}")

                    # 🔍 进行预测
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(df_transformed)[0]  # 获取 0 和 1 的概率
                        final_prediction = 1 if proba[1] > 0.5 else 0  # 如果 1 的概率大于 50%，预测为 1，否则为 0
                        predictions[model_name] = {
                            "prediction": final_prediction,  # 预测类别
                            "probabilities": {
                                "0": round(proba[0], 4),
                                "1": round(proba[1], 4)
                            }
                        }
                    else:
                        predictions[model_name] = int(
                            model.predict(df_transformed)[0])  # 继续返回整数（不支持 `predict_proba` 的模型）
                except Exception as e:
                    predictions[model_name] = f"❌ 预测失败: {str(e)}"
                    print(f"⚠️ `{model_name}` 预测时发生错误: {e}")

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
