from flask import Flask, render_template, jsonify, Response, request
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from mnist_model import MNISTNet
import plotly.graph_objects as go
import json
import time
import base64
import io
from PIL import Image
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# 檢查可用設備
available_devices = ['cpu']
if torch.xpu.is_available():
    available_devices.append('xpu')
    print("Intel GPU (XPU) 可用")
    device = torch.device('xpu')
else:
    print("Intel GPU (XPU) 不可用")
    device = torch.device('cpu')

if torch.cuda.is_available():
    available_devices.append('cuda')
    print("NVIDIA GPU (CUDA) 可用")

print(f"默認使用: {device}")

# 數據加載和預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters())

training_losses = []
test_accuracies = []

# 追踪訓練狀態的全局變量
current_batch = 0
total_batches = 0
current_loss = 0
is_training = False
training_start_time = None
training_time = 0

# 存儲當前批次圖像的全局變量
current_images = []
current_predictions = []
current_targets = []

# 添加模型保存路徑
MODEL_DIR = 'saved_models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 在全局變量部分添加
first_epoch = True

@app.route('/get_devices')
def get_devices():
    return jsonify({'devices': available_devices})

def tensor_to_base64(tensor):
    # 將tensor轉換為PIL圖像
    img = tensor.cpu().numpy() * 255
    img = img.astype(np.uint8).reshape(28, 28)
    pil_img = Image.fromarray(img)
    
    # 將PIL圖像轉換為base64字符串
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train_status')
def train_status():
    global current_batch, total_batches, current_loss, is_training
    global current_images, current_predictions, current_targets, training_time
    
    # 準備圖像數據
    images_data = []
    if len(current_images) > 0:
        for i in range(min(16, len(current_images))):
            img_base64 = tensor_to_base64(current_images[i])
            images_data.append({
                'image': img_base64,
                'prediction': int(current_predictions[i]) if len(current_predictions) > 0 else -1,
                'target': int(current_targets[i]) if len(current_targets) > 0 else -1
            })
    
    return jsonify({
        'current_batch': current_batch,
        'total_batches': total_batches,
        'current_loss': current_loss,
        'is_training': is_training,
        'images': images_data,
        'training_time': training_time
    })

@app.route('/train_epoch')
def train_epoch():
    global current_batch, total_batches, current_loss, is_training
    global current_images, current_predictions, current_targets
    global training_start_time, training_time, device, first_epoch
    
    try:
        device_name = request.args.get('device', 'cpu')
        print(f"嘗試切換到設備: {device_name}")
        
        device = torch.device(device_name)
        model.to(device)
        
        global optimizer
        optimizer = optim.Adam(model.parameters())
        
        print(f"成功切換到設備: {device}")
    except Exception as e:
        print(f"切換到 {device_name} 時發生錯誤: {str(e)}")
        device = torch.device('cpu')
        model.to(device)
        optimizer = optim.Adam(model.parameters())
    
    is_training = True
    training_start_time = time.time()
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    
    print(f"開始訓練，使用設備: {device}")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        current_batch = batch_idx + 1
        try:
            # 確保數據在正確的設備上
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # 更新當前批次的圖像和預測
            current_images = data.squeeze()
            current_predictions = output.argmax(dim=1)
            current_targets = target
            
            # 更新訓練時間
            training_time = time.time() - training_start_time
            
            time.sleep(0.1)
        except Exception as e:
            print(f"訓練過程中發生錯誤: {str(e)}")
            break
    
    avg_loss = total_loss / len(train_loader)
    training_losses.append(avg_loss)
    
    # 計算測試準確率
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)
    
    # 創建損失和準確率圖表
    loss_fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(training_losses) + 1)),  # 添加 x 軸數據
        y=training_losses,
        mode='lines+markers',  # 添加標記點
        name='Training Loss'
    ))
    
    acc_fig = go.Figure(data=go.Scatter(
        x=list(range(1, len(test_accuracies) + 1)),  # 添加 x 軸數據
        y=test_accuracies,
        mode='lines+markers',  # 添加標記點
        name='Test Accuracy'
    ))
    
    # 添加圖表標題和軸標籤
    loss_fig.update_layout(
        title='訓練損失',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        showlegend=True,
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,  # 設置 x 軸刻度間隔
            range=[0.5, max(1, len(training_losses)) + 0.5]  # 設置 x 軸範圍
        )
    )
    
    acc_fig.update_layout(
        title='測試準確率',
        xaxis_title='Epoch',
        yaxis_title='Accuracy (%)',
        showlegend=True,
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,  # 設置 x 軸刻度間隔
            range=[0.5, max(1, len(test_accuracies)) + 0.5]  # 設置 x 軸範圍
        )
    )
    
    # 如果是第一個 Epoch，確保圖表顯示範圍合適
    if first_epoch:
        loss_fig.update_yaxes(range=[0, training_losses[0] * 1.5])
        acc_fig.update_yaxes(range=[0, 100])
        first_epoch = False
    
    is_training = False
    current_batch = 0
    
    # 在訓練完成後保存模型
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODEL_DIR, f'mnist_model_{timestamp}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")
    except Exception as e:
        print(f"保存模型時發生錯誤: {str(e)}")
    
    return jsonify({
        'loss': avg_loss,
        'accuracy': accuracy,
        'loss_plot': loss_fig.to_json(),
        'acc_plot': acc_fig.to_json(),
        'first_epoch': first_epoch
    })

@app.route('/save_model')
def save_model():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODEL_DIR, f'mnist_model_{timestamp}.pth')
        torch.save(model.state_dict(), model_path)
        return jsonify({'success': True, 'path': model_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_latest_model')
def get_latest_model():
    try:
        models = os.listdir(MODEL_DIR)
        if not models:
            return jsonify({'success': False, 'error': 'No saved models found'})
        latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)))
        return jsonify({'success': True, 'path': latest_model})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 獲取繪圖數據
        data = request.json['image_data']
        # 將base64圖像數據轉換為PIL圖像
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # 預處理圖像
        image = image.convert('L')  # 轉換為灰度圖
        image = image.resize((28, 28))  # 調整大小為MNIST格式
        
        # 轉換為tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # 進行預測
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
            
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)