classdef BackendCNN_LSTM < handle
    % 后台CNN-LSTM网络，用于离线提取振动信号的深层时空特征
    % 输入：时间窗口（512点） → 输出：未来误差预测（标量）
    
    properties (SetAccess = private)
        net             % 训练好的SeriesNetwork或DAGNetwork
        inputSize      % 输入序列长度（时间步数）
        outputSize     % 输出维度（1）
        opts           % 训练选项
    end
    
    methods
        function obj = BackendCNN_LSTM(varargin)
            % 构造函数，可传入网络结构参数
            p = inputParser;
            addParameter(p, 'inputSize', 512);
            addParameter(p, 'outputSize', 1);
            addParameter(p, 'cnnFilters', [32, 64]);
            addParameter(p, 'cnnKernels', [8, 4]);
            addParameter(p, 'lstmUnits', 128);
            addParameter(p, 'dropout', 0.3);
            parse(p, varargin{:});
            
            obj.inputSize = p.Results.inputSize;
            obj.outputSize = p.Results.outputSize;
            
            % 构建网络层
            layers = [
                sequenceInputLayer(1, 'Name', 'input')   % 输入维度为1（标量时间序列）
                
                convolution1dLayer(p.Results.cnnKernels(1), p.Results.cnnFilters(1), ...
                    'Stride', 2, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                
                convolution1dLayer(p.Results.cnnKernels(2), p.Results.cnnFilters(2), ...
                    'Stride', 2, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                
                lstmLayer(p.Results.lstmUnits, 'OutputMode', 'last', 'Name', 'lstm')
                dropoutLayer(p.Results.dropout, 'Name', 'dropout')
                
                fullyConnectedLayer(64, 'Name', 'fc1')
                reluLayer('Name', 'relu_fc1')
                fullyConnectedLayer(32, 'Name', 'fc2')
                reluLayer('Name', 'relu_fc2')
                fullyConnectedLayer(obj.outputSize, 'Name', 'fc_out')
                regressionLayer('Name', 'output')
            ];
            
            obj.net = layerGraph(layers);
            % 设置默认训练选项（用户可覆盖）
            obj.opts = trainingOptions('adam', ...
                'MaxEpochs', 100, ...
                'MiniBatchSize', 64, ...
                'InitialLearnRate', 0.001, ...
                'Shuffle', 'every-epoch', ...
                'Plots', 'training-progress', ...
                'Verbose', true);
        end
        
        function train(obj, X, Y, varargin)
            % 训练网络
            % 输入：
            %   X - 训练数据，可为：
            %       1) 细胞数组，每个元素为时间序列向量（长度 inputSize x 1）
            %       2) 矩阵，尺寸为 [样本数 x inputSize]（每一行为一个序列）
            %   Y - 目标值，列向量，长度 = 样本数
            %   varargin - 可覆盖训练选项
            if nargin > 2
                obj.opts = trainingOptions(varargin{:});
            end
            
            % 将输入转换为细胞数组格式（sequenceInputLayer要求）
            if ~iscell(X)
                X = mat2cell(X, ones(size(X,1),1), size(X,2));
            end
            Y = Y(:);
            
            % 训练
            obj.net = trainNetwork(X, Y, obj.net, obj.opts);
        end
        
        function Y_pred = predict(obj, X)
            % 预测
            if ~iscell(X)
                X = mat2cell(X, ones(size(X,1),1), size(X,2));
            end
            Y_pred = predict(obj.net, X);
            Y_pred = Y_pred(:);
        end
        
        function saveModel(obj, filename)
            % 保存网络到.mat文件
            net = obj.net;
            save(filename, 'net');
        end
        
        function loadModel(obj, filename)
            % 加载预训练网络
            S = load(filename);
            obj.net = S.net;
        end
        
        function setTrainingOptions(obj, opts)
            obj.opts = opts;
        end
    end
end