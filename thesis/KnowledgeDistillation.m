classdef KnowledgeDistillation < handle
    % 知识蒸馏：将教师模型（CNN-LSTM）的知识迁移到学生模型（FLANN+RNN）
    % 支持回归任务的在线/离线蒸馏
    
    properties
        teacherNet          % 教师网络（BackendCNN_LSTM 或任何支持predict的对象）
        studentNet          % 学生网络（需提供前向传播和参数更新函数）
        temperature         % 温度参数（软化概率分布）
        lambda              % 硬标签损失权重（1-lambda为蒸馏损失权重）
        optimizer           % 学生网络优化器配置
    end
    
    methods
        function obj = KnowledgeDistillation(teacher, student, varargin)
            % 构造函数
            % 输入：
            %   teacher - 教师模型对象，必须有 predict(X) 方法
            %   student - 学生模型对象，必须有 forward(x) 和 update(gradients) 方法
            %   varargin - 'Temperature', 'Lambda', 'LearningRate'等
            obj.teacherNet = teacher;
            obj.studentNet = student;
            
            p = inputParser;
            addParameter(p, 'Temperature', 5.0);
            addParameter(p, 'Lambda', 0.3);
            addParameter(p, 'LearningRate', 0.001);
            parse(p, varargin{:});
            
            obj.temperature = p.Results.Temperature;
            obj.lambda = p.Results.Lambda;
            obj.optimizer.learningRate = p.Results.LearningRate;
        end
        
        function [total_loss, hard_loss, distill_loss] = distillationLoss(obj, x, y_true)
            % 计算蒸馏损失
            % 输入：
            %   x     - 单个样本输入（学生网络输入格式）
            %   y_true- 真实标签（标量）
            % 输出：
            %   各项损失值
            % 教师模型预测（软化）
            y_teacher = obj.teacherNet.predict(x);   % 标量
            soft_teacher = y_teacher / obj.temperature;
            
            % 学生模型前向
            y_student = obj.studentNet.forward(x);
            soft_student = y_student / obj.temperature;
            
            % 蒸馏损失（MSE between soft outputs）
            distill_loss = (soft_student - soft_teacher)^2;
            
            % 硬标签损失（MSE with true label）
            hard_loss = (y_student - y_true)^2;
            
            % 总损失
            total_loss = obj.lambda * hard_loss + (1 - obj.lambda) * distill_loss;
        end
        
        function distillOffline(obj, X, Y, numEpochs, batchSize)
            % 离线批量蒸馏：用教师网络指导训练学生网络
            % X, Y 为训练数据（格式同教师网络输入）
            n = size(X, 1);
            for epoch = 1:numEpochs
                % 随机打乱
                idx = randperm(n);
                X = X(idx, :);
                Y = Y(idx);
                
                total_loss_epoch = 0;
                for i = 1:batchSize:n
                    batch_end = min(i+batchSize-1, n);
                    X_batch = X(i:batch_end, :);
                    Y_batch = Y(i:batch_end);
                    
                    % 计算批量梯度（这里假设学生网络支持批量梯度下降）
                    % 简化：逐样本更新（可根据需要改为批量）
                    for j = 1:size(X_batch,1)
                        loss = obj.distillationLoss(X_batch(j,:), Y_batch(j));
                        total_loss_epoch = total_loss_epoch + loss;
                        % 计算学生网络梯度（需学生网络实现backward）
                        % 此处为学生网络留出接口，实际使用时需根据学生网络结构编写
                        % gradients = obj.studentNet.backward(loss);
                        % obj.studentNet.update(gradients, obj.optimizer.learningRate);
                    end
                end
                if mod(epoch, 10) == 0
                    fprintf('Epoch %d, Average Loss: %.6f\n', epoch, total_loss_epoch/n);
                end
            end
        end
        
        function [total_loss, grad] = distillationLossWithGrad(obj, x, y_true)
            % 返回损失和梯度（用于在线学习）
            % 需要学生网络支持自动微分（此处为概念性实现）
            % 实际使用时，可调用dlgradient实现端到端梯度
            y_teacher = obj.teacherNet.predict(x);
            soft_teacher = y_teacher / obj.temperature;
            
            y_student = obj.studentNet.forward(x);
            soft_student = y_student / obj.temperature;
            
            % 定义损失表达式
            hard_loss = (y_student - y_true)^2;
            distill_loss = (soft_student - soft_teacher)^2;
            total_loss = obj.lambda * hard_loss + (1 - obj.lambda) * distill_loss;
            
            % 计算梯度（此处为示意，实际需使用dlgradient）
            % grad = dlgradient(total_loss, obj.studentNet.parameters);
            grad = 0; % 占位
        end
    end
end