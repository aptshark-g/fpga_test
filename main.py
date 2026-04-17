import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ====================== GPU 配置 (关闭了惹祸的 AMP) ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前使用计算设备: {device} (已强制开启全精度 Float32 训练以保障梯度安全)")

# ==========================================
# 1. 鲁棒数据加载与清洗 (解决超级离群点陷阱)
# ==========================================
print("\n正在加载 MATLAB 数据 training_data.mat...")
mat_data = scipy.io.loadmat('s/training_data.mat')
features = mat_data['data_collect']['features'][0][0]  # [N, 19]
targets = mat_data['data_collect']['targets'][0][0].reshape(-1, 1)  # [N, 1]


# 【核心救命代码】Robust Scaler：剔除 1% 极端瞬态毛刺
def robust_normalize(data, name="Data"):
    p01 = np.percentile(data, 1, axis=0)
    p99 = np.percentile(data, 99, axis=0)
    clipped_data = np.clip(data, p01, p99)

    mean = np.mean(clipped_data, axis=0)
    std = np.std(clipped_data, axis=0) + 1e-8
    norm_data = (clipped_data - mean) / std

    print(f"[{name}] 体检 - 原最大值: {np.max(data):.4f} -> 截断后最大值: {np.max(clipped_data):.4f}")
    return norm_data, mean, std


X_norm, X_mean, X_std = robust_normalize(features, "输入特征 19 维")
Y_norm, target_mean, target_std = robust_normalize(targets, "目标电压")

# ==========================================
# 2. 构建 19通道 的真实滑动时间窗口
# ==========================================
SEQ_LEN = 16
X_seq, Y_seq = [], []

for i in range(len(X_norm) - SEQ_LEN):
    window = X_norm[i: i + SEQ_LEN, :]
    X_seq.append(window.T)  # [19, 16]
    Y_seq.append(Y_norm[i + SEQ_LEN - 1])

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32).view(-1, 1)

dataloader = DataLoader(
    TensorDataset(X_tensor, Y_tensor),
    batch_size=256,
    shuffle=True,
    pin_memory=True,
    num_workers=0
)


# ==========================================
# 3. Res-CNN-LSTM 残差融合网络
# ==========================================
class ResMultivariateCNNLSTM(nn.Module):
    def __init__(self):
        super(ResMultivariateCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(19, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.ReLU()

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.skip = nn.Linear(19, 1)

    def forward(self, x_in):
        x = self.act1(self.bn1(self.conv1(x_in)))
        x = self.act2(self.bn2(self.conv2(x)))

        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        main_out = self.fc(out)

        last_step_features = x_in[:, :, -1]
        skip_out = self.skip(last_step_features)
        return main_out + skip_out


model = ResMultivariateCNNLSTM().to(device)
criterion = nn.MSELoss().to(device)
# 提升学习率，因为数据已经彻底干净了
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ==========================================
# 4. 全精度安全训练
# ==========================================
epochs = 50
print(f"\n🚀 开始云端全精度训练 (总Epochs: {epochs})...")
loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_Y = batch_Y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 彻底抛弃 AMP，用纯32位浮点数计算，杜绝梯度下溢
        outputs = model(batch_X)
        batch_Y = batch_Y.view_as(outputs)
        loss = criterion(outputs, batch_Y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], MSE Loss: {avg_loss:.6f}')

# ==========================================
# 5. 预测与导出
# ==========================================
model.eval()
preds_list = []
chunk_size = 2048

with torch.no_grad():
    for i in range(0, len(X_tensor), chunk_size):
        chunk_X = X_tensor[i: i + chunk_size].to(device)
        chunk_pred = model(chunk_X)
        preds_list.append(chunk_pred.cpu().numpy())

preds_norm_np = np.vstack(preds_list)
Y_numpy = Y_tensor.numpy()

target_var = np.var(Y_numpy)
mse = np.mean((Y_numpy - preds_norm_np) ** 2)
r2_score = 1 - mse / target_var
print(f"\n🎉 教师模型全局拟合度 R² = {r2_score:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', label='Train Loss')
plt.title('Training Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(Y_numpy[1000:1200], 'k-', label='True Target', alpha=0.7)
plt.plot(preds_norm_np[1000:1200], 'r--', label='Model Prediction', alpha=0.8)
plt.title('Prediction vs True')
plt.legend()
plt.tight_layout()
plt.show()

soft_labels_real = preds_norm_np * target_std + target_mean
export_data = {
    'soft_labels': soft_labels_real,
    'X_mean': X_mean,
    'X_std': X_std
}
scipy.io.savemat('teacher_soft_labels.mat', export_data)
print("\n✅ 软标签已成功导出！")