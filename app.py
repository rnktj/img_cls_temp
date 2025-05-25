# 画像分類アプリのテンプレコード（PyTorch + Streamlit）

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st

# クラス名（例：犬と猫）
CLASS_NAMES = ["cat", "dog"]

# モデルの準備（ResNet18の転移学習済みモデル）
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    # ここでは仮にランダム初期化されたモデルを使う（学習済みモデルがあればload_state_dict）
    model.eval()
    return model

# 前処理定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 推論関数
def predict(image: Image.Image, model: nn.Module):
    image = transform(image).unsqueeze(0)  # バッチ次元を追加
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]

# Streamlit UI
st.title("画像分類デモ（PyTorch + Streamlit）")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='アップロード画像', use_column_width=True)

    model = load_model()
    prediction = predict(image, model)
    st.success(f"予測結果: {prediction}")
