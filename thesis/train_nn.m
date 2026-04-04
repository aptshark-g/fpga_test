clear; clc;
load('training_data.mat');   % 包含 data_collect

% 特征和目标
X = data_collect.features;
y = data_collect.targets(:);

% 剔除异常值
mu_y = mean(y); std_y = std(y);
outlier = abs(y - mu_y) > 3*std_y;
X(outlier, :) = [];
y(outlier) = [];
fprintf('剔除 %d 个异常值，剩余 %d 个样本\n', sum(outlier), length(y));

% 划分训练/验证集
n = size(X,1);
idx = randperm(n);
train_idx = idx(1:round(0.8*n));
val_idx = idx(round(0.8*n)+1:end);
X_train = X(train_idx,:);
y_train = y(train_idx);
X_val = X(val_idx,:);
y_val = y(val_idx);

% 标准化
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma==0) = 1;
X_train_norm = (X_train - mu) ./ sigma;
X_val_norm = (X_val - mu) ./ sigma;

% 初始化网络（小容量）
input_dim = 19;
hidden_dims = [32, 16];
nn = ResidualNN_Simple(input_dim, hidden_dims);
% 设置标准化参数（使用训练集的统计量）
nn.feature_mean = mu(:);
nn.feature_std = sigma(:);

% 训练参数
epochs = 2000;
lr = 0.001;
batch_size = 64;
best_val_loss = inf;
patience = 100;
no_improve = 0;
val_loss_history = [];

% 将数据转为列向量格式（便于矩阵运算）
X_train_t = X_train_norm';   % input_dim × N
y_train_t = y_train';        % 1 × N
N_train = size(X_train_t,2);

for epoch = 1:epochs
    % 随机打乱批量顺序
    idx_epoch = randperm(N_train);
    X_batch = X_train_t(:, idx_epoch);
    y_batch = y_train_t(idx_epoch);
    
    % 小批量训练
    for batch = 1:batch_size:N_train
        batch_end = min(batch+batch_size-1, N_train);
        Xb = X_batch(:, batch:batch_end);
        yb = y_batch(batch:batch_end);
        
        % 前向传播
        Z1 = nn.W1 * Xb + nn.b1;
        A1 = max(0, Z1);
        Z2 = nn.W2 * A1 + nn.b2;
        A2 = max(0, Z2);
        Z3 = nn.W3 * A2 + nn.b3;
        Y = tanh(Z3);
        
        % 损失梯度
        dY = 2 * (Y - yb) / size(Xb,2);
        dZ3 = dY .* (1 - Y.^2);
        dW3 = dZ3 * A2';
        db3 = sum(dZ3, 2);
        
        dA2 = nn.W3' * dZ3;
        dZ2 = dA2 .* (A2 > 0);
        dW2 = dZ2 * A1';
        db2 = sum(dZ2, 2);
        
        dA1 = nn.W2' * dZ2;
        dZ1 = dA1 .* (A1 > 0);
        dW1 = dZ1 * Xb';
        db1 = sum(dZ1, 2);
        
        % 更新权重
        nn.W3 = nn.W3 - lr * dW3;
        nn.b3 = nn.b3 - lr * db3;
        nn.W2 = nn.W2 - lr * dW2;
        nn.b2 = nn.b2 - lr * db2;
        nn.W1 = nn.W1 - lr * dW1;
        nn.b1 = nn.b1 - lr * db1;
    end
    
    % 每10个epoch评估验证损失
    if mod(epoch, 10) == 0
        % 验证集前向
        X_val_t = X_val_norm';
        Z1v = nn.W1 * X_val_t + nn.b1;
        A1v = max(0, Z1v);
        Z2v = nn.W2 * A1v + nn.b2;
        A2v = max(0, Z2v);
        Z3v = nn.W3 * A2v + nn.b3;
        Yv = tanh(Z3v);
        val_loss = mean((Yv - y_val').^2);
        val_loss_history(end+1) = val_loss;
        
        if val_loss < best_val_loss
            best_val_loss = val_loss;
            no_improve = 0;
            best_nn = nn;   % 保存最佳模型
        else
            no_improve = no_improve + 1;
            if no_improve >= patience
                fprintf('早停于 epoch %d\n', epoch);
                break;
            end
        end
        fprintf('Epoch %d, Val Loss = %.6f\n', epoch, val_loss);
    end
    
    % 学习率衰减
    if mod(epoch, 500) == 0
        lr = lr * 0.5;
        fprintf('学习率衰减至 %.6f\n', lr);
    end
end

% 使用最佳模型
nn = best_nn;
save('trained_nn_improved.mat', 'nn');
fprintf('训练完成，最佳验证损失 = %.6f\n', best_val_loss);

% 绘制验证损失曲线
figure;
plot(10:10:10*length(val_loss_history), val_loss_history, 'b-o');
xlabel('Epoch'); ylabel('验证损失 (MSE)');
title('神经网络训练曲线');
grid on;