classdef HybridController < handle
    % 混合控制器类：线性 FxLMS + 神经网络补偿器，带自适应混合系数和状态机
    
    properties
        alpha               % 当前混合系数 (0~alpha_max)
        alpha_min           % 最小混合系数
        alpha_max           % 最大混合系数
        current_state       % 当前状态：'pure_linear', 'hybrid_trial', 'hybrid_stable'
        
        % 性能评估相关
        performance_window  % 性能评估窗口长度（步数）
        performance_history % 存储误差信号的历史值（用于计算 RMS）
        baseline_performance % 纯线性控制时的基准性能（RMS 误差）
        performance_improvement_threshold = 0.15; % 性能提升阈值（15%）
        performance_degradation_threshold = 0.20; % 性能退化阈值（20%）
        
        % 混合系数调整参数
        alpha_inc_rate = 0.001;    % 稳定模式下 α 增加速率（每步）
        alpha_dec_rate = 0.002;    % 试验模式下 α 减少速率（可选）
        
        % 状态机参数
        trial_length = 1000;       % 试验模式持续时间（步数）
        trial_step_counter = 0;    % 试验模式计数器
        
        % 其他
        verbose = false;           % 是否打印状态切换信息
    end
    
    methods
        function obj = HybridController(alpha_min, alpha_max, performance_window)
            % 构造函数
            % alpha_min: 最小混合系数（默认 0）
            % alpha_max: 最大混合系数（默认 0.8）
            % performance_window: 性能评估窗口长度（默认 100）
            
            if nargin < 1, alpha_min = 0; end
            if nargin < 2, alpha_max = 0.8; end
            if nargin < 3, performance_window = 100; end
            
            obj.alpha_min = alpha_min;
            obj.alpha_max = alpha_max;
            obj.alpha = alpha_min;   % 初始为纯线性模式
            obj.current_state = 'pure_linear';
            obj.performance_window = performance_window;
            obj.performance_history = [];
            obj.baseline_performance = inf; % 初始化为无穷大
        end
        
        function y_total = combine(obj, y_linear, y_nn)
            % 混合控制输出
            % y_linear: 线性控制器输出（标量）
            % y_nn: 神经网络补偿输出（标量）
            y_total = y_linear + obj.alpha * y_nn;
        end
        
        function adapt_alpha(obj, error_signal, reference_signal)
            % 自适应调整混合系数（需在每个仿真步调用）
            % error_signal: 当前误差（残余振动）
            % reference_signal: 参考信号（可选，用于性能评估）
            
            % 更新性能历史
            obj.performance_history = [obj.performance_history; abs(error_signal)];
            if length(obj.performance_history) > obj.performance_window
                obj.performance_history = obj.performance_history(end-obj.performance_window+1:end);
            end
            
            % 计算当前性能（RMS 误差）
            current_performance = rms(obj.performance_history);
            
            % 如果是纯线性模式且尚未记录基准性能，记录当前性能作为基准
            if strcmp(obj.current_state, 'pure_linear') && obj.baseline_performance == inf
                obj.baseline_performance = current_performance;
                if obj.verbose
                    fprintf('基准性能已记录: RMS = %.4f\n', obj.baseline_performance);
                end
            end
            
            % 状态机逻辑
            switch obj.current_state
                case 'pure_linear'
                    % 若性能持续低于阈值（即误差较大），尝试切换到混合模式
                    % 这里简单判断：如果当前性能比基准性能差超过 50%，则触发切换
                    if current_performance > obj.baseline_performance * 1.5
                        obj.transition_to('hybrid_trial');
                    end
                    
                    
                case 'hybrid_trial'
                    obj.trial_step_counter = obj.trial_step_counter + 1;
    
                    % 逐步增加 α（从 0 开始，最多到 alpha_max/2）
                    if obj.alpha < obj.alpha_max * 0.5
                        obj.alpha = min(obj.alpha_max * 0.5, obj.alpha + obj.alpha_inc_rate);
                    end
    
                    % 达到试验长度后，评估性能
                    if obj.trial_step_counter >= obj.trial_length
                    % 计算试验期间的平均性能
                        len_hist = length(obj.performance_history);
                        if len_hist >= obj.trial_length
                            trial_performance = rms(obj.performance_history(end-obj.trial_length+1:end));
                        else
                            trial_performance = rms(obj.performance_history); % 使用全部可用历史
                        end
                        % 如果性能提升超过阈值，切换到稳定混合模式
                        if trial_performance < obj.baseline_performance * (1 - obj.performance_improvement_threshold)
                            obj.transition_to('hybrid_stable');
                        else
                        % 否则，回到纯线性模式，α 归零
                            obj.transition_to('pure_linear');
                        end
                    end

                    
                case 'hybrid_stable'
                    % 稳定混合模式：微调 α，但保持在一定范围内
                    % 如果当前性能优于基准，可尝试略微增加 α
                    if current_performance < obj.baseline_performance
                        obj.alpha = min(obj.alpha_max, obj.alpha + obj.alpha_inc_rate);
                    else
                        % 如果性能变差，减小 α
                        obj.alpha = max(obj.alpha_min, obj.alpha - obj.alpha_inc_rate);
                    end
                    
                    % 如果性能严重退化（超过基准 20%），切换回纯线性模式
                    if current_performance > obj.baseline_performance * (1 + obj.performance_degradation_threshold)
                        obj.transition_to('pure_linear');
                    end
            end
            
            % 确保 α 在合法范围内
            obj.alpha = max(obj.alpha_min, min(obj.alpha_max, obj.alpha));
        end
        
        function transition_to(obj, new_state)
            % 状态切换方法
            old_state = obj.current_state;
            obj.current_state = new_state;
            
            if obj.verbose
                %fprintf('状态切换: %s -> %s\n', old_state, new_state);
            end
            
            switch new_state
                case 'pure_linear'
                    obj.alpha = obj.alpha_min;
                    obj.trial_step_counter = 0;
                    % 重置性能基准（允许重新学习）
                    % 注意：这里不重置 baseline_performance，以保持对比
                case 'hybrid_trial'
                    obj.alpha = obj.alpha_min;
                    obj.trial_step_counter = 0;
                case 'hybrid_stable'
                    % 记录稳定混合模式下的 α 作为起始点
                    % α 已在进入前设置好，不做额外操作
            end
        end
        
        function reset(obj)
            % 重置控制器状态（用于多次仿真）
            obj.alpha = obj.alpha_min;
            obj.current_state = 'pure_linear';
            obj.performance_history = [];
            obj.baseline_performance = inf;
            obj.trial_step_counter = 0;
        end
    end
end