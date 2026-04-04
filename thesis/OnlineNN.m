classdef OnlineNN < handle
    % 在线神经网络类（基于手写网络 + 经验回放）
    
    properties
        net                 % ResidualNN_Simple 实例
        buffer_size         % 最大缓冲区长度
        experience_buffer   % 存储 {features, residual} 的元胞数组
        buffer_idx          % 当前写入位置（循环覆盖）
        buffer_count        % 实际存储的样本数
        update_interval     % 每隔多少步更新一次网络
        step_counter        % 总步数计数器
        batch_size          % 每次训练的小批量大小
        train_epochs        % 每次更新的训练轮数
        learning_rate       % 学习率
    end
    
    methods
        function obj = OnlineNN(input_dim, hidden_dims, buffer_size, update_interval, batch_size, train_epochs, lr)
            % 构造函数
            % input_dim: 输入特征维度（默认19）
            % hidden_dims: 隐藏层神经元数向量（默认[12,8]）
            % buffer_size: 经验缓冲区大小（默认1000）
            % update_interval: 更新间隔步数（默认100）
            % batch_size: 每次训练的批量大小（默认32）
            % train_epochs: 每次更新的训练轮数（默认5）
            % lr: 学习率（默认0.001）
            
            if nargin < 1, input_dim = 19; end
            if nargin < 2, hidden_dims = [12, 8]; end
            if nargin < 3, buffer_size = 1000; end
            if nargin < 4, update_interval = 100; end
            if nargin < 5, batch_size = 32; end
            if nargin < 6, train_epochs = 5; end
            if nargin < 7, lr = 0.001; end
            
            obj.net = ResidualNN_Simple(input_dim, hidden_dims);
            obj.buffer_size = buffer_size;
            obj.update_interval = update_interval;
            obj.batch_size = batch_size;
            obj.train_epochs = train_epochs;
            obj.learning_rate = lr;
            
            % 初始化缓冲区
            obj.experience_buffer = cell(buffer_size, 2);
            obj.buffer_idx = 1;
            obj.buffer_count = 0;
            obj.step_counter = 0;
        end
        
        function y = forward(obj, features)
            % 前向推理（调用内部网络）
            % features: 1×input_dim 行向量
            y = obj.net.forward(features');
        end
        
        function store_experience(obj, features, residual)
            % 存储经验（features, residual）
            % features: 1×input_dim 行向量
            % residual: 标量（目标与线性输出的差）
            obj.experience_buffer{obj.buffer_idx, 1} = features;
            obj.experience_buffer{obj.buffer_idx, 2} = residual;
            obj.buffer_idx = mod(obj.buffer_idx, obj.buffer_size) + 1;
            obj.buffer_count = min(obj.buffer_count + 1, obj.buffer_size);
            obj.step_counter = obj.step_counter + 1;
            
            % 定期更新网络
            if mod(obj.step_counter, obj.update_interval) == 0 && obj.buffer_count >= obj.batch_size
                obj.update_network();
            end
        end
        
        function update_network(obj)
            % 从缓冲区随机采样一批数据，训练网络（微调）
            % 采样数量 = min(batch_size, buffer_count)
            n_samples = min(obj.batch_size, obj.buffer_count);
            indices = randperm(obj.buffer_count, n_samples);
            
            % 收集采样数据
            X = zeros(n_samples, size(obj.experience_buffer{1,1}, 2));
            Y = zeros(n_samples, 1);
            for i = 1:n_samples
                X(i,:) = obj.experience_buffer{indices(i), 1};
                Y(i) = obj.experience_buffer{indices(i), 2};
            end
            
            % 训练网络（使用当前标准化参数，但注意标准化参数最好也在线更新）
            % 这里简单使用当前网络的标准化参数（离线训练时已设置，或可动态更新）
            % 注意：为了稳定，通常在线训练时不更新标准化参数，或者用全局的均值和方差。
            obj.net.train_residual(X, zeros(n_samples,1), Y, obj.train_epochs, obj.learning_rate);
        end
        
        function set_normalization(obj, mean_vec, std_vec)
            % 设置标准化参数（可从离线数据计算）
            % mean_vec: input_dim×1 向量
            % std_vec:  input_dim×1 向量
            obj.net.feature_mean = mean_vec;
            obj.net.feature_std = std_vec;
        end
    end
end