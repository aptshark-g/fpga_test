% 结构动力学模型类
classdef StructuralDynamics < handle
    properties
        % 模态参数 [基于曾威(2006)实验模态分析]
        modal_freqs = [128, 335, 790];           % 模态频率 [Hz]
        modal_damping = [0.02, 0.03, 0.05];      % 模态阻尼比
        
        % 模态振型矩阵（3个位置×3个模态）
        % 行1: 作动器位置，行2: 传感器位置，行3: 加工点位置
        modal_shapes = [1.0, 0.8, 0.5;          % 作动器位置模态
                        0.7, 1.0, 0.3;          % 传感器位置模态
                        0.5, 0.6, 1.0];         % 加工点位置模态
        
        % 扩展参数（基于顶刊[10]的实测数据）
        coupling_frequency_shift = -8;          % 耦合模态频率偏移 [Hz]
        load_damping_curve = [5, 0.025; 20, 0.04]; % 负载-阻尼关系 [N, 阻尼比]
        
        % 状态空间表示
        A = [];        % 状态矩阵（离散）
        B = [];        % 输入矩阵（离散）
        C = [];        % 输出矩阵（离散）
        D = [];        % 前馈矩阵（离散）
        
        % 当前状态变量
        x = [];        % 状态变量
        
        % 工作参数
        current_load = 5;    % 当前负载 [N]
        enable_coupling_effect = false; % 是否启用耦合效应
        sample_rate = 10000; % 采样率 [Hz]
    end
    
    methods
        function obj = StructuralDynamics(sample_rate, load, enable_coupling)
            % 构造状态空间模型并离散化
            % 输入: 
            %   sample_rate - 采样率 [Hz]
            %   load - 负载 [N] (可选)
            %   enable_coupling - 是否启用耦合频率偏移 (可选)
            
            if nargin >= 1
                obj.sample_rate = sample_rate;
            end
            if nargin >= 2
                obj.current_load = load;
            end
            if nargin >= 3
                obj.enable_coupling_effect = enable_coupling;
            end
            
            % 根据负载调整阻尼
            obj.update_damping_for_load();
            
            % 根据耦合效应调整频率
            obj.update_frequency_for_coupling();
            
            % 构建连续时间状态空间模型
            [Ac, Bc, Cc, Dc] = obj.build_continuous_model();
            
            % 离散化
            dt = 1/obj.sample_rate;
            [obj.A, obj.B, obj.C, obj.D] = obj.discretize_model(Ac, Bc, Cc, Dc, dt);
            
            % 初始化状态变量
            n_states = size(obj.A, 1);
            obj.x = zeros(n_states, 1);
        end
        
        function update_damping_for_load(obj, new_load)
        % 根据负载更新阻尼参数
        % 输入: new_load - 新的负载值 [N] (可选)
    
        if nargin >= 2
            obj.current_load = new_load;
        end
    
        % 原始模态阻尼比（来自表格）
        original_damping = [0.02, 0.03, 0.05];
        original_avg = mean(original_damping);
    
        % 根据负载插值计算目标平均阻尼比
        loads = obj.load_damping_curve(:, 1);
        target_dampings = obj.load_damping_curve(:, 2);
    
        if obj.current_load <= loads(1)
            target_avg = target_dampings(1);
        elseif obj.current_load >= loads(end)
            target_avg = target_dampings(end);
        else
            target_avg = interp1(loads, target_dampings, obj.current_load, 'linear');
        end
    
        % 调整各模态阻尼比，保持相对比例不变
        scaling_factor = target_avg / original_avg;
        obj.modal_damping = original_damping * scaling_factor;
    
        fprintf('负载 %.1f N: 目标平均阻尼比 = %.4f\n', obj.current_load, target_avg);
        fprintf('  各模态阻尼比: [%.4f, %.4f, %.4f]\n', ...
            obj.modal_damping(1), obj.modal_damping(2), obj.modal_damping(3));
        end
        
        function update_frequency_for_coupling(obj, enable_coupling)
            % 根据耦合效应调整频率
            % 输入: enable_coupling - 是否启用耦合频率偏移 (可选)
            
            if nargin >= 2
                obj.enable_coupling_effect = enable_coupling;
            end
            
            if obj.enable_coupling_effect
                % 应用耦合频率偏移
                obj.modal_freqs = obj.modal_freqs + obj.coupling_frequency_shift;
                fprintf('启用耦合效应: 模态频率偏移 -8 Hz\n');
                fprintf('新模态频率: [%.1f, %.1f, %.1f] Hz\n', ...
                    obj.modal_freqs(1), obj.modal_freqs(2), obj.modal_freqs(3));
            else
                % 恢复原始频率
                obj.modal_freqs = [128, 335, 790];
            end
        end
        
        function [Ac, Bc, Cc, Dc] = build_continuous_model(obj)
            % 构建连续时间状态空间模型
            % 输出: Ac, Bc, Cc, Dc - 连续时间状态空间矩阵
            
            n_modes = length(obj.modal_freqs);
            n_states = 2 * n_modes;
            
            % 构建对角化的模态状态矩阵
            Ac = zeros(n_states);
            for i = 1:n_modes
                omega = 2 * pi * obj.modal_freqs(i);        % 自然频率 [rad/s]
                zeta = obj.modal_damping(i);                % 阻尼比
                
                % 第i个模态的状态矩阵块
                idx = 2*(i-1) + (1:2);
                Ac(idx, idx) = [0, 1; -omega^2, -2*zeta*omega];
            end
            
            % 输入矩阵Bc (力输入到作动器位置)
            Bc = zeros(n_states, 1);
            for i = 1:n_modes
                idx = 2*(i-1) + 2;  % 速度状态的位置
                Bc(idx) = obj.modal_shapes(1, i);  % 作动器位置的模态振型
            end
            
            % 输出矩阵Cc (三个位置的位移输出)
            Cc = zeros(3, n_states);
            for i = 1:n_modes
                col = 2*(i-1) + 1;  % 位移状态的位置
                
                % 作动器位置位移
                Cc(1, col) = obj.modal_shapes(1, i);
                
                % 传感器位置位移
                Cc(2, col) = obj.modal_shapes(2, i);
                
                % 加工点位置位移
                Cc(3, col) = obj.modal_shapes(3, i);
            end
            
            % 前馈矩阵Dc
            Dc = zeros(3, 1);
        end
        
        function [Ad, Bd, Cd, Dd] = discretize_model(obj, Ac, Bc, Cc, Dc, dt)
            % 使用零阶保持法离散化状态空间模型
            % 输入: Ac, Bc, Cc, Dc - 连续时间矩阵, dt - 采样时间
            % 输出: Ad, Bd, Cd, Dd - 离散时间矩阵
            
            n_states = size(Ac, 1);
            
            % 使用矩阵指数法离散化
            M = expm([Ac, Bc; zeros(1, n_states+1)] * dt);
            Ad = M(1:n_states, 1:n_states);
            Bd = M(1:n_states, n_states+1:end);
            Cd = Cc;
            Dd = Dc;
        end
        
        function [displacements, velocities] = respond(obj, force, dt)
            % 结构对作用力的响应计算
            % 输入: force - 作用力 [N], dt - 时间步长 [s] (可选)
            % 输出: displacements - 各位置位移 [m]
            %       velocities - 各位置速度 [m/s]
            
            if nargin < 3
                dt = 1/obj.sample_rate;
            end
            
            % 如果dt变化，需要重新离散化
            current_dt = 1/obj.sample_rate;
            if abs(dt - current_dt) > 1e-10
                % 更新采样率并重新离散化
                obj.sample_rate = 1/dt;
                [Ac, Bc, Cc, Dc] = obj.build_continuous_model();
                [obj.A, obj.B, obj.C, obj.D] = obj.discretize_model(Ac, Bc, Cc, Dc, dt);
            end
            
            % 状态更新
            obj.x = obj.A * obj.x + obj.B * force;
            
            % 计算输出
            outputs = obj.C * obj.x + obj.D * force;
            
            % 位移输出
            displacements = outputs(1:3);  % 三个位置的位移
            
            % 如果需要速度，可以从状态变量中提取
            if nargout >= 2
                n_modes = length(obj.modal_freqs);
                velocities = zeros(3, 1);
                for i = 1:n_modes
                    idx = 2*(i-1) + 2;  % 速度状态的位置
                    for j = 1:3
                        velocities(j) = velocities(j) + obj.modal_shapes(j, i) * obj.x(idx);
                    end
                end
            end
        end
        
        function acceleration = get_acceleration(obj, position_index)
            % 获取指定位置的加速度
            % 输入: position_index - 位置索引 (1:作动器, 2:传感器, 3:加工点)
            % 输出: acceleration - 加速度 [m/s²]
            
            n_modes = length(obj.modal_freqs);
            acceleration = 0;
            
            for i = 1:n_modes
                omega = 2 * pi * obj.modal_freqs(i);        % 自然频率 [rad/s]
                zeta = obj.modal_damping(i);                % 阻尼比
                
                % 状态索引
                q_idx = 2*(i-1) + 1;  % 位移
                dq_idx = 2*(i-1) + 2; % 速度
                
                % 加速度贡献
                modal_acc = -omega^2 * obj.x(q_idx) - 2*zeta*omega * obj.x(dq_idx);
                acceleration = acceleration + obj.modal_shapes(position_index, i) * modal_acc;
            end
        end
        
        function [freq_response, freqs] = frequency_response(obj, freqs, output_index, input_index)
            % 计算频响函数
            % 输入: freqs - 频率数组 [Hz], output_index - 输出位置索引
            %       input_index - 输入位置索引（目前只支持1个输入）
            % 输出: freq_response - 频响函数值
            
            if nargin < 3
                output_index = 2;  % 默认传感器位置
            end
            if nargin < 4
                input_index = 1;   % 默认作动器输入
            end
            
            n_freqs = length(freqs);
            freq_response = zeros(n_freqs, 1);
            
            [Ac, Bc, Cc, Dc] = obj.build_continuous_model();
            
            for i = 1:n_freqs
                s = 1i * 2 * pi * freqs(i);
                H = Cc(output_index, :) * inv(s*eye(size(Ac)) - Ac) * Bc(:, input_index) + Dc(output_index, input_index);
                freq_response(i) = abs(H);
            end
        end
        
        function reset(obj)
            % 重置结构状态
            obj.x = zeros(size(obj.x));
        end
        
        function plot_frequency_response(obj, freqs)
            % 绘制频响函数
            % 输入: freqs - 频率数组 [Hz] (可选)
            
            if nargin < 2
                freqs = logspace(1, 3, 500);  % 10-1000 Hz
            end
            
            figure('Name', '结构频响函数', 'Position', [100, 100, 800, 600]);
            
            % 计算三个位置的频响
            H_actuator = obj.frequency_response(freqs, 1);
            H_sensor = obj.frequency_response(freqs, 2);
            H_tool = obj.frequency_response(freqs, 3);
            
            subplot(2,1,1);
            semilogy(freqs, H_actuator, 'b-', 'LineWidth', 1.5);
            hold on;
            semilogy(freqs, H_sensor, 'r-', 'LineWidth', 1.5);
            semilogy(freqs, H_tool, 'g-', 'LineWidth', 1.5);
            
            % 标记模态频率
            for i = 1:length(obj.modal_freqs)
                line([obj.modal_freqs(i), obj.modal_freqs(i)], ylim, ...
                    'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
                text(obj.modal_freqs(i), max(ylim)*0.9, ...
                    sprintf('f%d=%.0fHz', i, obj.modal_freqs(i)), ...
                    'HorizontalAlignment', 'center');
            end
            
            title('结构频响函数（位移/力）');
            xlabel('频率 [Hz]');
            ylabel('幅值 [m/N]');
            legend('作动器位置', '传感器位置', '加工点位置', 'Location', 'best');
            grid on;
            xlim([0, 1000]);
            
            subplot(2,1,2);
            plot(freqs, 20*log10(H_sensor ./ H_actuator), 'k-', 'LineWidth', 1.5);
            title('传感器/作动器传递函数');
            xlabel('频率 [Hz]');
            ylabel('增益 [dB]');
            grid on;
            xlim([0, 1000]);
        end
        
        function plot_mode_shapes(obj)
            % 绘制模态振型
            
            figure('Name', '模态振型', 'Position', [100, 100, 1200, 400]);
            
            positions = {'作动器', '传感器', '加工点'};
            
            for i = 1:length(obj.modal_freqs)
                subplot(1, length(obj.modal_freqs), i);
                
                % 绘制三个位置的振型值
                bar(1:3, obj.modal_shapes(:, i));
                
                title(sprintf('模态%d (%.0f Hz)', i, obj.modal_freqs(i)));
                xlabel('位置');
                ylabel('振型幅值');
                set(gca, 'XTick', 1:3, 'XTickLabel', positions);
                grid on;
                ylim([0, 1.1]);
            end
            
            sgtitle('结构模态振型');
        end
    end
end