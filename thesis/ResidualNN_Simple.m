classdef ResidualNN_Simple < handle
    % 纯 MATLAB 实现的轻量级学生网络（端侧部署专版）
    % 已移除输出层 tanh 限幅，完美支持全量程电压回归
    
    properties
        W1, b1, W2, b2, W3, b3   % 权重和偏置
        input_dim, hidden1, hidden2
        feature_mean, feature_std
    end
    
    methods
        function obj = ResidualNN_Simple(input_dim, hidden_dims)
            obj.input_dim = input_dim;
            obj.hidden1 = hidden_dims(1);
            obj.hidden2 = hidden_dims(2);
            
            % Xavier 初始化 (保证初始梯度稳定)
            obj.W1 = randn(obj.hidden1, input_dim) * sqrt(2 / input_dim);
            obj.b1 = zeros(obj.hidden1, 1);
            obj.W2 = randn(obj.hidden2, obj.hidden1) * sqrt(2 / obj.hidden1);
            obj.b2 = zeros(obj.hidden2, 1);
            obj.W3 = randn(1, obj.hidden2) * sqrt(2 / obj.hidden2);
            obj.b3 = 0;
            
            obj.feature_mean = zeros(input_dim, 1);
            obj.feature_std = ones(input_dim, 1);
        end
        
        function y = forward(obj, x)
            % 在线实时前向推理 (加入 Leaky ReLU 保持活性)
            x_norm = (x - obj.feature_mean) ./ obj.feature_std;
            
            % 隐藏层 1 (Leaky ReLU)
            Z1 = obj.W1 * x_norm + obj.b1;
            h1 = max(0.01 * Z1, Z1); 
            
            % 隐藏层 2 (Leaky ReLU)
            Z2 = obj.W2 * h1 + obj.b2;
            h2 = max(0.01 * Z2, Z2); 
            
            % 输出层 (纯线性)
            y = obj.W3 * h2 + obj.b3;
        end
        
        function train_distillation(obj, features, y_target_blended, epochs, lr)
            % 蒸馏训练：必须同步使用 Leaky ReLU 的梯度
            N = size(features, 1);
            obj.feature_mean = mean(features, 1)';
            obj.feature_std = std(features, 0, 1)';
            obj.feature_std(obj.feature_std == 0) = 1;
            X_full = ((features - obj.feature_mean') ./ obj.feature_std')';   
            T_full = y_target_blended(:)';  

            % 使用 Mini-Batch 提升收敛稳定性
            batch_size = 4096; num_batches = floor(N / batch_size);
            for epoch = 1:epochs
                idx = randperm(N); X_shuf = X_full(:, idx); T_shuf = T_full(:, idx);
                for b = 1:num_batches
                    X = X_shuf(:, (b-1)*batch_size+1 : b*batch_size);
                    T = T_shuf(:, (b-1)*batch_size+1 : b*batch_size);
                    
                    % 前向传播
                    Z1 = obj.W1 * X + obj.b1; A1 = max(0.01 * Z1, Z1);
                    Z2 = obj.W2 * A1 + obj.b2; A2 = max(0.01 * Z2, Z2);
                    Y = obj.W3 * A2 + obj.b3;
                    
                    % 反向传播
                    dY = 2 * (Y - T) / batch_size;
                    dZ3 = dY;
                    dW3 = dZ3 * A2'; db3 = sum(dZ3, 2);
                    dA2 = obj.W3' * dZ3;
                    dZ2 = dA2 .* ((Z2 > 0) + 0.01 * (Z2 <= 0)); % Leaky 梯度
                    dW2 = dZ2 * A1'; db2 = sum(dZ2, 2);
                    dA1 = obj.W2' * dZ2;
                    dZ1 = dA1 .* ((Z1 > 0) + 0.01 * (Z1 <= 0)); % Leaky 梯度
                    dW1 = dZ1 * X'; db1 = sum(dZ1, 2);
                    
                    % 梯度裁剪防止 NaN
                    gnorm = norm([dW1(:); db1(:); dW2(:); db2(:); dW3(:); db3(:)]);
                    if gnorm > 1.0, clip = 1.0/gnorm; 
                        dW1=dW1*clip; dW2=dW2*clip; dW3=dW3*clip; 
                    end
                    
                    % 更新
                    obj.W3 = obj.W3 - lr * dW3; obj.b3 = obj.b3 - lr * db3;
                    obj.W2 = obj.W2 - lr * dW2; obj.b2 = obj.b2 - lr * db2;
                    obj.W1 = obj.W1 - lr * dW1; obj.b1 = obj.b1 - lr * db1;
                end
                if mod(epoch, 500) == 0, fprintf('Epoch %d, Loss: %.6f\n', epoch, mean((Y-T).^2)); end
            end
        end

        
        function train_residual(obj, X, linear_out, target, epochs, lr)
    % 在线残差学习：仅更新输出层权重（W3, b3），隐藏层冻结
    % 符合低算力在线微调需求，与蒸馏训练（冻结特征提取层）逻辑一致
    % X: n_samples × input_dim 特征矩阵
    % linear_out: n_samples × 1 （FxLMS 输出，保留接口但本方法未使用）
    % target: n_samples × 1 （目标补偿电压）
    % epochs: 训练轮数
    % lr: 学习率
    
    N = size(X, 1);
    if N == 0
        return;
    end
    
    % 标准化输入（使用蒸馏时固化好的均值和标准差）
    X_norm = (X - obj.feature_mean') ./ obj.feature_std';
    
    for epoch = 1:epochs
        % 前向传播到隐藏层2（不更新隐藏层权重）
        Z1 = obj.W1 * X_norm' + obj.b1;   % hidden1 × N
        A1 = max(0, Z1);                  % ReLU
        Z2 = obj.W2 * A1 + obj.b2;        % hidden2 × N
        A2 = max(0, Z2);                  % ReLU
        
        % 输出层预测
        Y = obj.W3 * A2 + obj.b3;          % 1 × N
        Y = Y(:);                          % N × 1
        
        % 计算误差
        error = target(:) - Y;             % N × 1
        
        % 仅更新输出层权重（LMS 规则）
        obj.W3 = obj.W3 + lr * (error' * A2');
        obj.b3 = obj.b3 + lr * sum(error);
        
        % 可选：每10个epoch打印一次损失（调试用）
        if mod(epoch, 10) == 0 && epoch <= 50
            loss = mean(error.^2);
            fprintf('train_residual epoch %d, MSE = %.6f\n', epoch, loss);
        end
    end
end
    end
end

