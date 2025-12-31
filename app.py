import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 🆕 LOAD ENV & DATABASE
from dotenv import load_dotenv
load_dotenv()
from database import init_db, save_training_record

# ================= APP INIT =================
app = Flask(__name__)

# 🆕 INIT DATABASE
try:
    init_db()
except Exception as e:
    print("⚠️ Database not available yet:", e)

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ================= HOME =================
@app.route("/")
def home():
    return render_template("home.html")

# ================= TRAIN PAGE =================
@app.route("/train")
def train():
    return render_template("train.html")

# ================= TRAIN PROCESS =================
@app.route("/train_process", methods=["POST"])
def train_process():
    file = request.files["csv_file"]

    # 🆕 READ CSV BYTES FOR DATABASE
    csv_bytes = file.read()
    file.seek(0)

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path).dropna()

    target = df.columns[-1]
    if not np.issubdtype(df[target].dtype, np.number):
        return "❌ Last column must be numeric"

    X = df.drop(columns=[target])
    y = df[target]

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    accuracy = round(r2_score(y_test, model.predict(X_test)), 3)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(encoders, open("encoders.pkl", "wb"))
    pickle.dump(list(X.columns), open("features.pkl", "wb"))

    # 🆕 SAVE TRAINING RECORD
    save_training_record(
        file_name=file.filename,
        csv_bytes=csv_bytes,
        rows=df.shape[0],
        columns=df.shape[1],
        target=target,
        features=",".join(list(X.columns)),
        accuracy=float(accuracy)
    )

    buffer = StringIO()
    df.info(buf=buffer)
    dataset_info = buffer.getvalue()

    numeric_df = df.select_dtypes(include="number")

    # ================= GRAPHS =================

    plt.hist(y, bins=20)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/target_dist.png")
    plt.close()

    plt.plot(y.values)
    plt.title("Target Trend")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/train_line.png")
    plt.close()

    X.mean().plot(kind="bar")
    plt.title("Feature Mean Comparison")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/train_bar.png")
    plt.close()

    if numeric_df.shape[1] >= 2:
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{PLOT_FOLDER}/train_heatmap.png")
        plt.close()

    pd.qcut(y, q=3, labels=["Low", "Medium", "High"]).value_counts().plot(
        kind="pie", autopct="%1.1f%%"
    )
    plt.title("Target Category Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/train_pie.png")
    plt.close()

    if numeric_df.shape[1] >= 2:
        col = numeric_df.columns[0]
        plt.scatter(df[col], y)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title("Feature vs Target")
        plt.tight_layout()
        plt.savefig(f"{PLOT_FOLDER}/train_scatter.png")
        plt.close()

    return render_template(
        "train_result.html",
        rows=df.shape[0],
        cols=df.shape[1],
        target=target,
        accuracy=accuracy,
        head_table=df.head().to_html(index=False),
        tail_table=df.tail().to_html(index=False),
        describe_table=df.describe().to_html(),
        dataset_info=dataset_info,
        features=list(X.columns),
        x_train_shape=X_train.shape,
        x_test_shape=X_test.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape
    )

# ================= DETECT PAGE =================
@app.route("/detect")
def detect():
    features = pickle.load(open("features.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    dropdowns = {k: list(v.classes_) for k, v in encoders.items()}
    return render_template("detect.html", features=features, dropdowns=dropdowns)

# ================= DETECT PROCESS =================
@app.route("/detect_process", methods=["POST"])
def detect_process():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))

    inputs = {}
    values = []

    for f in features:
        raw = request.form[f]
        inputs[f] = raw
        if f in encoders:
            raw = encoders[f].transform([raw])[0]
        else:
            raw = float(raw)
        values.append(raw)

    values_df = pd.DataFrame([values], columns=features)
    prediction = model.predict(scaler.transform(values_df))[0]

    risk_score = round(max(0, min(abs(prediction) * 100, 100)), 2)
    risk_level = (
        "Low Risk" if risk_score < 33 else
        "Medium Risk" if risk_score < 66 else
        "High Risk"
    )

    plt.bar(["Risk Score"], [risk_score])
    plt.ylim(0, 100)
    plt.title("Risk Score (%)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/det_bar.png")
    plt.close()

    plt.pie([risk_score, 100-risk_score], labels=["Risk", "Safe"], autopct="%1.1f%%")
    plt.title("Risk Composition")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/det_pie.png")
    plt.close()

    trend_x = ["Baseline", "Predicted"]
    trend_y = [0, risk_score]

    plt.plot(trend_x, trend_y, marker="o")
    plt.ylim(0, 100)
    plt.ylabel("Risk Score (%)")
    plt.title("Financial Risk Trend Analysis")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/det_line.png")
    plt.close()

    plt.scatter(range(len(values)), values)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.title("Input Feature Scatter")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/det_scatter.png")
    plt.close()

    sns.heatmap(values_df, annot=True, cmap="coolwarm")
    plt.title("Input Feature Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOT_FOLDER}/det_heatmap.png")
    plt.close()

    return render_template(
        "detect_result.html",
        prediction=round(prediction, 3),
        risk_score=risk_score,
        risk_level=risk_level,
        inputs=inputs
    )

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



















# import matplotlib
# matplotlib.use("Agg")

# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# import os, pickle
# import matplotlib.pyplot as plt
# import seaborn as sns

# from io import StringIO
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score

# # ================= APP INIT =================
# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# PLOT_FOLDER = "static/plots"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PLOT_FOLDER, exist_ok=True)

# # ================= HOME =================
# @app.route("/")
# def home():
#     return render_template("home.html")

# # ================= TRAIN PAGE =================
# @app.route("/train")
# def train():
#     return render_template("train.html")

# # ================= TRAIN PROCESS =================
# @app.route("/train_process", methods=["POST"])
# def train_process():
#     file = request.files["csv_file"]
#     path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(path)

#     df = pd.read_csv(path).dropna()

#     # Target = last column
#     target = df.columns[-1]
#     if not np.issubdtype(df[target].dtype, np.number):
#         return "❌ Last column must be numeric"

#     X = df.drop(columns=[target])
#     y = df[target]

#     # Encode categorical columns
#     encoders = {}
#     for col in X.select_dtypes(include="object").columns:
#         le = LabelEncoder()
#         X[col] = le.fit_transform(X[col].astype(str))
#         encoders[col] = le

#     # Scale
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     # Model
#     model = RandomForestRegressor(random_state=42)
#     model.fit(X_train, y_train)

#     accuracy = round(r2_score(y_test, model.predict(X_test)), 3)

#     # Save artifacts
#     pickle.dump(model, open("model.pkl", "wb"))
#     pickle.dump(scaler, open("scaler.pkl", "wb"))
#     pickle.dump(encoders, open("encoders.pkl", "wb"))
#     pickle.dump(list(X.columns), open("features.pkl", "wb"))

#     # ================= DATASET INFO =================
#     buffer = StringIO()
#     df.info(buf=buffer)
#     dataset_info = buffer.getvalue()

#     numeric_df = df.select_dtypes(include="number")

#     # ================= GRAPHS =================

#     # Target Distribution
#     plt.hist(y, bins=20)
#     plt.title("Target Distribution")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/target_dist.png")
#     plt.close()

#     # Target Trend
#     plt.plot(y.values)
#     plt.title("Target Trend")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/train_line.png")
#     plt.close()

#     # Feature Mean Bar
#     X.mean().plot(kind="bar")
#     plt.title("Feature Mean Comparison")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/train_bar.png")
#     plt.close()

#     # Heatmap
#     if numeric_df.shape[1] >= 2:
#         sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
#         plt.title("Correlation Heatmap")
#         plt.tight_layout()
#         plt.savefig(f"{PLOT_FOLDER}/train_heatmap.png")
#         plt.close()

#     # Pie
#     pd.qcut(y, q=3, labels=["Low", "Medium", "High"]).value_counts().plot(
#         kind="pie", autopct="%1.1f%%"
#     )
#     plt.title("Target Category Distribution")
#     plt.ylabel("")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/train_pie.png")
#     plt.close()

#     # Scatter
#     if numeric_df.shape[1] >= 2:
#         col = numeric_df.columns[0]
#         plt.scatter(df[col], y)
#         plt.xlabel(col)
#         plt.ylabel(target)
#         plt.title("Feature vs Target")
#         plt.tight_layout()
#         plt.savefig(f"{PLOT_FOLDER}/train_scatter.png")
#         plt.close()

#     return render_template(
#         "train_result.html",
#         rows=df.shape[0],
#         cols=df.shape[1],
#         target=target,
#         accuracy=accuracy,
#         head_table=df.head().to_html(index=False),
#         tail_table=df.tail().to_html(index=False),
#         describe_table=df.describe().to_html(),
#         dataset_info=dataset_info,
#         features=list(X.columns),
#         x_train_shape=X_train.shape,
#         x_test_shape=X_test.shape,
#         y_train_shape=y_train.shape,
#         y_test_shape=y_test.shape
#     )

# # ================= DETECT PAGE =================
# @app.route("/detect")
# def detect():
#     features = pickle.load(open("features.pkl", "rb"))
#     encoders = pickle.load(open("encoders.pkl", "rb"))
#     dropdowns = {k: list(v.classes_) for k, v in encoders.items()}
#     return render_template("detect.html", features=features, dropdowns=dropdowns)

# # ================= DETECT PROCESS =================
# @app.route("/detect_process", methods=["POST"])
# def detect_process():
#     model = pickle.load(open("model.pkl", "rb"))
#     scaler = pickle.load(open("scaler.pkl", "rb"))
#     encoders = pickle.load(open("encoders.pkl", "rb"))
#     features = pickle.load(open("features.pkl", "rb"))

#     inputs = {}
#     values = []

#     for f in features:
#         raw = request.form[f]
#         inputs[f] = raw
#         if f in encoders:
#             raw = encoders[f].transform([raw])[0]
#         else:
#             raw = float(raw)
#         values.append(raw)

#     # FIX: use DataFrame to match scaler training
#     values_df = pd.DataFrame([values], columns=features)
#     prediction = model.predict(scaler.transform(values_df))[0]

#     risk_score = round(max(0, min(abs(prediction) * 100, 100)), 2)
#     risk_level = (
#         "Low Risk" if risk_score < 33 else
#         "Medium Risk" if risk_score < 66 else
#         "High Risk"
#     )

#     # ================= INDUSTRY-LEVEL VISUALS =================

#     # Risk Bar
#     plt.bar(["Risk Score"], [risk_score])
#     plt.ylim(0, 100)
#     plt.title("Risk Score (%)")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/det_bar.png")
#     plt.close()

#     # Risk Pie
#     plt.pie([risk_score, 100-risk_score], labels=["Risk", "Safe"], autopct="%1.1f%%")
#     plt.title("Risk Composition")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/det_pie.png")
#     plt.close()

#     # ===== INDUSTRY-LEVEL RISK TREND =====
#     trend_x = ["Baseline", "Predicted"]
#     trend_y = [0, risk_score]

#     plt.figure(figsize=(7, 4))
#     plt.plot(trend_x, trend_y, marker="o", linewidth=2)

#     plt.axhspan(0, 33, alpha=0.1)
#     plt.axhspan(33, 66, alpha=0.1)
#     plt.axhspan(66, 100, alpha=0.1)

#     plt.axhline(33, linestyle="--")
#     plt.axhline(66, linestyle="--")

#     plt.text(0.02, 16, "Low Risk Zone")
#     plt.text(0.02, 49, "Medium Risk Zone")
#     plt.text(0.02, 82, "High Risk Zone")

#     plt.ylim(0, 100)
#     plt.ylabel("Risk Score (%)")
#     plt.title("Financial Risk Trend Analysis")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/det_line.png")
#     plt.close()

#     # Input Scatter
#     plt.scatter(range(len(values)), values)
#     plt.xlabel("Feature Index")
#     plt.ylabel("Feature Value")
#     plt.title("Input Feature Scatter")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/det_scatter.png")
#     plt.close()

#     # Heatmap
#     sns.heatmap(values_df, annot=True, cmap="coolwarm")
#     plt.title("Input Feature Heatmap")
#     plt.tight_layout()
#     plt.savefig(f"{PLOT_FOLDER}/det_heatmap.png")
#     plt.close()

#     return render_template(
#         "detect_result.html",
#         prediction=round(prediction, 3),
#         risk_score=risk_score,
#         risk_level=risk_level,
#         inputs=inputs
#     )

# # ================= RUN =================
# if __name__ == "__main__":
#     app.run(debug=True)
