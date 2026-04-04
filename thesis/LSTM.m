classdef LSTM < handle
    % 长短期记忆网络层（手动实现，支持单层单方向）
    properties
        hiddenSize      % 隐藏单元数
        inputSize       % 输入维度
        Wf, Wi, Wo, Wc  % 遗忘门、输入门、输出门、候选状态权重 [hiddenSize, inputSize]
        Uf, Ui, Uo, Uc  % 循环权重 [hiddenSize, hiddenSize]
        bf, bi, bo, bc  % 偏置 [hiddenSize, 1]
        % 缓存变量用于反向传播
        cache
    end
    
    methods
        function obj = LSTM(hiddenSize, inputSize)
            obj.hiddenSize = hiddenSize;
            obj.inputSize = inputSize;
            % 初始化（Xavier）
            scale = sqrt(2 / (hiddenSize + inputSize));
            obj.Wf = randn(hiddenSize, inputSize) * scale;
            obj.Wi = randn(hiddenSize, inputSize) * scale;
            obj.Wo = randn(hiddenSize, inputSize) * scale;
            obj.Wc = randn(hiddenSize, inputSize) * scale;
            obj.Uf = randn(hiddenSize, hiddenSize) * scale;
            obj.Ui = randn(hiddenSize, hiddenSize) * scale;
            obj.Uo = randn(hiddenSize, hiddenSize) * scale;
            obj.Uc = randn(hiddenSize, hiddenSize) * scale;
            obj.bf = zeros(hiddenSize, 1);
            obj.bi = zeros(hiddenSize, 1);
            obj.bo = zeros(hiddenSize, 1);
            obj.bc = zeros(hiddenSize, 1);
            obj.cache = struct();
        end
        
        function [h, c] = forward(obj, x, h_prev, c_prev)
            % x: [inputSize, 1] 当前时刻输入
            % h_prev, c_prev: [hiddenSize, 1] 上一时刻状态
            % 返回 h, c
            % 计算门
            f = sigmoid(obj.Wf * x + obj.Uf * h_prev + obj.bf);
            i = sigmoid(obj.Wi * x + obj.Ui * h_prev + obj.bi);
            o = sigmoid(obj.Wo * x + obj.Uo * h_prev + obj.bo);
            c_tilde = tanh(obj.Wc * x + obj.Uc * h_prev + obj.bc);
            c = f .* c_prev + i .* c_tilde;
            h = o .* tanh(c);
            % 缓存
            obj.cache.x = x;
            obj.cache.h_prev = h_prev;
            obj.cache.c_prev = c_prev;
            obj.cache.f = f;
            obj.cache.i = i;
            obj.cache.o = o;
            obj.cache.c_tilde = c_tilde;
            obj.cache.c = c;
            obj.cache.h = h;
        end
        
        function [dx, dh_prev, dc_prev] = backward(obj, dh, dc_next, lr)
            % dh: 从上一层传来的 h 梯度 [hiddenSize,1]
            % dc_next: 下一时刻的 c 梯度（初始为0）
            % 返回 dx (输入梯度), dh_prev, dc_prev
            x = obj.cache.x;
            h_prev = obj.cache.h_prev;
            c_prev = obj.cache.c_prev;
            f = obj.cache.f;
            i = obj.cache.i;
            o = obj.cache.o;
            c_tilde = obj.cache.c_tilde;
            c = obj.cache.c;
            h = obj.cache.h;
            
            % 输出门梯度
            do = dh .* tanh(c);
            do = do .* o .* (1 - o);
            % 细胞状态梯度
            dc = dh .* o .* (1 - tanh(c).^2) + dc_next;
            % 候选细胞梯度
            dc_tilde = dc .* i .* (1 - c_tilde.^2);
            di = dc .* c_tilde .* i .* (1 - i);
            df = dc .* c_prev .* f .* (1 - f);
            
            % 输入梯度
            dx = obj.Wf' * df + obj.Wi' * di + obj.Wo' * do + obj.Wc' * dc_tilde;
            % 上一时刻隐藏状态梯度
            dh_prev = obj.Uf' * df + obj.Ui' * di + obj.Uo' * do + obj.Uc' * dc_tilde;
            % 上一时刻细胞状态梯度
            dc_prev = dc .* f;
            
            % 权重梯度
            obj.Wf = obj.Wf - lr * (df * x');
            obj.Wi = obj.Wi - lr * (di * x');
            obj.Wo = obj.Wo - lr * (do * x');
            obj.Wc = obj.Wc - lr * (dc_tilde * x');
            obj.Uf = obj.Uf - lr * (df * h_prev');
            obj.Ui = obj.Ui - lr * (di * h_prev');
            obj.Uo = obj.Uo - lr * (do * h_prev');
            obj.Uc = obj.Uc - lr * (dc_tilde * h_prev');
            obj.bf = obj.bf - lr * df;
            obj.bi = obj.bi - lr * di;
            obj.bo = obj.bo - lr * do;
            obj.bc = obj.bc - lr * dc_tilde;
        end
    end
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end