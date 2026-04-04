classdef Conv1D < handle
    % 一维卷积层（手动实现，无工具箱依赖）
    properties
        filters         % 卷积核数量（输出通道数）
        kernelSize      % 卷积核大小
        stride          % 步长
        padding         % 填充模式：'same' 或 'valid'
        weights         % 卷积核权重 [filters, inputChannels, kernelSize]
        bias            % 偏置 [filters, 1]
        gradW           % 权重梯度
        gradb           % 偏置梯度
        inputCache      % 缓存输入用于反向传播
    end
    
    methods
        function obj = Conv1D(filters, kernelSize, stride, padding)
            obj.filters = filters;
            obj.kernelSize = kernelSize;
            obj.stride = stride;
            obj.padding = padding;
            % 权重初始化（Xavier）
            obj.weights = randn(filters, 1, kernelSize) * sqrt(2 / kernelSize);
            obj.bias = zeros(filters, 1);
        end
        
        function Y = forward(obj, X)
            % X: [inputChannels, inputLength]  (输入通道数, 时间步)
            % Y: [filters, outputLength]
            obj.inputCache = X;
            [inputChannels, L_in] = size(X);
            % 计算填充
            if strcmp(obj.padding, 'same')
                pad_total = max(0, (L_in - 1) * obj.stride + obj.kernelSize - L_in);
                pad_left = floor(pad_total / 2);
                pad_right = pad_total - pad_left;
                X_pad = [zeros(inputChannels, pad_left), X, zeros(inputChannels, pad_right)];
            else
                X_pad = X;
                pad_left = 0;
            end
            L_out = floor((size(X_pad,2) - obj.kernelSize) / obj.stride) + 1;
            Y = zeros(obj.filters, L_out);
            for f = 1:obj.filters
                for i = 1:L_out
                    start = (i-1)*obj.stride + 1;
                    end_idx = start + obj.kernelSize - 1;
                    x_win = X_pad(:, start:end_idx);  % [inputChannels, kernelSize]
                    % 卷积：权重与窗口点乘
                    y_val = sum(sum(obj.weights(f,1,:) .* x_win)) + obj.bias(f);
                    Y(f, i) = y_val;
                end
            end
        end
        
        function dX = backward(obj, dY, lr)
            % dY: [filters, L_out] 上一层传来的梯度
            % dX: 返回给前一层的梯度
            X = obj.inputCache;
            [inputChannels, L_in] = size(X);
            % 计算填充
            if strcmp(obj.padding, 'same')
                pad_total = max(0, (L_in - 1) * obj.stride + obj.kernelSize - L_in);
                pad_left = floor(pad_total / 2);
                pad_right = pad_total - pad_left;
                X_pad = [zeros(inputChannels, pad_left), X, zeros(inputChannels, pad_right)];
                dX_pad = zeros(size(X_pad));
            else
                X_pad = X;
                dX_pad = zeros(size(X));
                pad_left = 0;
            end
            L_out = size(dY,2);
            % 初始化梯度
            obj.gradW = zeros(size(obj.weights));
            obj.gradb = sum(dY, 2);
            for f = 1:obj.filters
                for i = 1:L_out
                    start = (i-1)*obj.stride + 1;
                    end_idx = start + obj.kernelSize - 1;
                    x_win = X_pad(:, start:end_idx);
                    dY_val = dY(f, i);
                    % 权重梯度
                    obj.gradW(f,1,:) = obj.gradW(f,1,:) + dY_val * x_win;
                    % 输入梯度
                    dX_pad(:, start:end_idx) = dX_pad(:, start:end_idx) + dY_val * squeeze(obj.weights(f,1,:));
                end
            end
            % 移除填充部分的梯度
            if strcmp(obj.padding, 'same')
                dX = dX_pad(:, pad_left+1:end-pad_right);
            else
                dX = dX_pad;
            end
            % 更新权重
            obj.weights = obj.weights - lr * obj.gradW;
            obj.bias = obj.bias - lr * obj.gradb;
        end
    end
end