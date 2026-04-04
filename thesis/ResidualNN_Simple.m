classdef ResidualNN_Simple < handle
    % 纯 MATLAB 实现的残差神经网络（无工具箱依赖）
    
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
            
            % Xavier 初始化
            obj.W1 = randn(obj.hidden1, input_dim) * sqrt(2 / input_dim);
            obj.b1 = zeros(obj.hidden1, 1);
            obj.W2 = randn(obj.hidden2, obj.hidden1) * sqrt(2 / obj.hidden1);
            obj.b2 = zeros(obj.hidden2, 1);
            obj.W3 = randn(1, obj.hidden2) * sqrt(2 / obj.hidden2);
            obj.b3 = 0;
            
            obj.feature_mean = zeros(input_dim, 1);
            obj.feature_std = ones(input_dim, 1);
        end
        % --原
        %function y = forward(obj, x)
            % x: 特征向量 (input_dim x 1) 或矩阵 (input_dim x N)
            %x_norm = (x - obj.feature_mean) ./ obj.feature_std;
            %h1 = max(0, obj.W1 * x_norm + obj.b1);
            %h2 = max(0, obj.W2 * h1 + obj.b2);
            %y = tanh(obj.W3 * h2 + obj.b3);
        %end

        function y = forward(obj, x)
            x_norm = (x - obj.feature_mean) ./ obj.feature_std;
            h1 = max(0, obj.W1 * x_norm + obj.b1);
            h1 = min(h1, 10);   % 裁剪到 [0,10]
            h2 = max(0, obj.W2 * h1 + obj.b2);
            h2 = min(h2, 10);   % 裁剪到 [0,10]
            y = tanh(obj.W3 * h2 + obj.b3);
        end
        
        function train_residual(obj, features, y_linear, y_target, epochs, lr)
            fprintf('features size: [%d, %d]\n', size(features,1), size(features,2));
            fprintf('y_linear size: [%d, %d]\n', size(y_linear,1), size(y_linear,2));
            fprintf('y_target size: [%d, %d]\n', size(y_target,1), size(y_target,2));
            % 输入：features N x input_dim, y_linear N x 1, y_target N x 1
            N = size(features, 1);
            input_dim = size(features, 2);
    
            % 标准化
            obj.feature_mean = mean(features, 1)';
            obj.feature_std = std(features, 0, 1)';
            obj.feature_std(obj.feature_std == 0) = 1;
    
            % 标准化并转置为 input_dim x N
            X = ((features - obj.feature_mean') ./ obj.feature_std')';   % input_dim x N
            T = (y_target - y_linear)';  % 1×N
            T= T(:)';  % 确保是行向量
    
            if nargin < 6, lr = 0.001; end
    
            % 可选：打印尺寸确认
            fprintf('输入维度: X size = [%d, %d], T size = [%d, %d]\n', size(X,1), size(X,2), size(T,1), size(T,2));
    
            for epoch = 1:epochs
                % 前向传播
                Z1 = obj.W1 * X + obj.b1;   % hidden1 x N
                A1 = max(0, Z1);
                Z2 = obj.W2 * A1 + obj.b2;  % hidden2 x N
                A2 = max(0, Z2);
                Z3 = obj.W3 * A2 + obj.b3;  % 1 x N
                Y = tanh(Z3);
        
                % 损失
                loss = mean((Y - T).^2);
        
                % 反向传播
                dY = 2 * (Y - T) / N;           % 1 x N
                dZ3 = dY .* (1 - Y.^2);         % 1 x N
        
                % 检查维度
                assert(size(dZ3,1)==1 && size(dZ3,2)==N, 'dZ3 维度错误');
                assert(size(obj.W3,1)==1 && size(obj.W3,2)==obj.hidden2, 'W3 维度错误');
        
                dW3 = dZ3 * A2';                % 1 x hidden2
                db3 = sum(dZ3, 2);              % 1 x 1
        
                dA2 = obj.W3' * dZ3;            % hidden2 x N
                dZ2 = dA2 .* (A2 > 0);
                dW2 = dZ2 * A1';                % hidden2 x hidden1
                db2 = sum(dZ2, 2);
        
                dA1 = obj.W2' * dZ2;            % hidden1 x N
                dZ1 = dA1 .* (A1 > 0);
                dW1 = dZ1 * X';                 % hidden1 x input_dim
                db1 = sum(dZ1, 2);
        
                % 更新
                obj.W3 = obj.W3 - lr * dW3;
                obj.b3 = obj.b3 - lr * db3;
                obj.W2 = obj.W2 - lr * dW2;
                obj.b2 = obj.b2 - lr * db2;
                obj.W1 = obj.W1 - lr * dW1;
                obj.b1 = obj.b1 - lr * db1;
        
                if mod(epoch, 100) == 0
                    fprintf('Epoch %d, Loss: %.6f\n', epoch, loss);
                end
            end
        end
    end
end

function y = relu(x)
    y = max(0, x);
end