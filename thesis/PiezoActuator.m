% 压电作动器模型类
classdef PiezoActuator < handle
    properties
        % 线性参数 [基于相晖(2006)实测数据]
        K = 0.85;          % 静态增益 [μm/V]
        fn = 3100;         % 谐振频率 [Hz]
        zeta = 0.023;      % 阻尼比

        % 迟滞参数 (Bouc-Wen模型) [基于相晖(2006)响应曲线拟合]
        alpha = 0.15;
        beta = 0.035;
        gamma = 0.025;

        % 双频激励下的迟滞参数 [借鉴顶刊[10]双频迟滞特性数据]
        dual_freq_alpha = 0.17;   % 双频激励下的α参数
        dual_freq_beta = 0.04;    % 双频激励下的β参数

        % 状态变量
        hysteresis_state = 0;  % 迟滞状态h(t)
        prev_input = 0;        % 上一时刻输入
        current_mode = 'single_frequency'; % 当前工作模式

        % 数值积分方法选择
        integration_method = 'rk4'; % 'euler', 'rk2', 'rk4'

        % 离散状态空间模型（预计算）
        Ad          % 状态矩阵（离散）
        Bd          % 输入矩阵（离散）
        Cd          % 输出矩阵（离散）
        Dd          % 前馈矩阵（离散）
        state       % 状态变量
        dt          % 采样间隔（在构造函数中设置）
    end

    methods
        function obj = PiezoActuator(dt)
            % 构造函数：预计算离散状态空间模型
            % 输入: dt - 采样时间 [s] (可选，默认 1/10000)
            if nargin < 1
                dt = 1/10000;
            end
            obj.dt = dt;
            obj.compute_discrete_model();
            obj.reset();  % 重置状态
        end

        function compute_discrete_model(obj)
            % 计算离散状态空间矩阵（固定采样率）
            wn = 2 * pi * obj.fn;
            num = [obj.K];
            den = [1/wn^2, 2*obj.zeta/wn, 1];
            sys = tf(num, den);
            sysd = c2d(sys, obj.dt, 'zoh');
            [obj.Ad, obj.Bd, obj.Cd, obj.Dd] = ssdata(sysd);
            % 确保状态向量初始化为零
            obj.state = zeros(size(obj.Ad, 1), 1);
        end

        function set_mode(obj, mode)
            % 设置作动器工作模式
            % mode: 'single_frequency' - 单频激励模式
            %       'dual_frequency' - 双频激励模式
            obj.current_mode = mode;

            switch mode
                case 'dual_frequency'
                    % 双频激励模式下使用调整后的迟滞参数
                    obj.alpha = obj.dual_freq_alpha;
                    obj.beta = obj.dual_freq_beta;
                    disp('切换到双频激励模式：采用顶刊[10]迟滞参数');
                otherwise
                    % 单频激励模式下使用原始迟滞参数
                    obj.alpha = 0.15;
                    obj.beta = 0.035;
                    disp('切换到单频激励模式：采用相晖(2006)迟滞参数');
            end
        end

        function [displacement, force] = actuate(obj, voltage, dt)
            % 计算作动器输出，包含线性动力学和迟滞补偿
            % 输入: voltage - 控制电压 [V], dt - 时间步长 [s] (必须与构造函数中的dt一致)
            % 输出: displacement - 位移输出 [μm], force - 作用力 [N]

            % 如果dt与预计算的不一致，重新离散化（可选）
            if dt ~= obj.dt
                obj.dt = dt;
                obj.compute_discrete_model();
            end

            % 根据当前模式选择迟滞参数
            if strcmp(obj.current_mode, 'dual_frequency')
                alpha = obj.dual_freq_alpha;
                beta = obj.dual_freq_beta;
                gamma = obj.gamma;
            else
                alpha = obj.alpha;
                beta = obj.beta;
                gamma = obj.gamma;
            end

            % Bouc-Wen模型数值积分（采用RK4方法）
            if strcmp(obj.integration_method, 'rk4')
                k1 = dt * obj.bouc_wen_ode(obj.hysteresis_state, voltage, alpha, beta, gamma);
                k2 = dt * obj.bouc_wen_ode(obj.hysteresis_state + k1/2, voltage, alpha, beta, gamma);
                k3 = dt * obj.bouc_wen_ode(obj.hysteresis_state + k2/2, voltage, alpha, beta, gamma);
                k4 = dt * obj.bouc_wen_ode(obj.hysteresis_state + k3, voltage, alpha, beta, gamma);
                obj.hysteresis_state = obj.hysteresis_state + (k1 + 2*k2 + 2*k3 + k4)/6;
            else
                obj.hysteresis_state = obj.hysteresis_state + dt * ...
                    obj.bouc_wen_ode(obj.hysteresis_state, voltage, alpha, beta, gamma);
            end

            % 线性动力学部分：使用预计算的离散状态空间
            obj.state = obj.Ad * obj.state + obj.Bd * voltage;
            linear_response = obj.Cd * obj.state + obj.Dd * voltage;

            % 总位移 = 线性响应 + 迟滞补偿
            displacement = linear_response + obj.hysteresis_state;

            % 作用力计算（假设作动器刚度）
            stiffness = 1e6; % 刚度系数 [N/m]
            force = stiffness * displacement * 1e-6; % 位移单位转换为m

            % 更新上一时刻输入
            obj.prev_input = voltage;
        end

        function dh = bouc_wen_ode(obj, h, u, alpha, beta, gamma)
            % Bouc-Wen模型微分方程
            % dh/dt = α·du/dt - β·|du/dt|·h - γ·du/dt·|h|
            du = (u - obj.prev_input);
            if abs(du) < 1e-10
                dh = 0;
            else
                dh = alpha * du - beta * abs(du) * h - gamma * du * abs(h);
            end
        end

        function reset(obj)
            % 重置作动器状态
            obj.hysteresis_state = 0;
            obj.prev_input = 0;
            if ~isempty(obj.state)
                obj.state(:) = 0;
            end
            disp('作动器状态已重置');
        end

        function plot_hysteresis(obj, voltage_signal, dt, mode)
            % 绘制作动器迟滞环
            % 输入: voltage_signal - 电压信号数组, dt - 时间步长, mode - 工作模式
            if nargin < 4, mode = 'single_frequency'; end
            obj.set_mode(mode);
            obj.reset();

            n_samples = length(voltage_signal);
            displacement = zeros(n_samples, 1);
            force = zeros(n_samples, 1);

            for i = 1:n_samples
                [displacement(i), force(i)] = obj.actuate(voltage_signal(i), dt);
            end

            figure;
            plot(voltage_signal, displacement, 'b.', 'MarkerSize', 1);
            xlabel('控制电压 (V)');
            ylabel('位移输出 (\mum)');

            if strcmp(mode, 'dual_frequency')
                title(['双频激励模式迟滞环 (α=', num2str(obj.dual_freq_alpha), ...
                    ', β=', num2str(obj.dual_freq_beta), ')']);
            else
                title(['单频激励模式迟滞环 (α=', num2str(obj.alpha), ...
                    ', β=', num2str(obj.beta), ')']);
            end
            grid on;

            hysteresis_area = polyarea(voltage_signal, displacement);
            disp(['迟滞环面积: ', num2str(hysteresis_area), ' V·μm']);

            figure;
            subplot(2,1,1);
            plot((0:n_samples-1)*dt, voltage_signal, 'r', 'LineWidth', 1.5);
            xlabel('时间 (s)');
            ylabel('控制电压 (V)');
            title('作动器输入电压');
            grid on;

            subplot(2,1,2);
            plot((0:n_samples-1)*dt, displacement, 'b', 'LineWidth', 1.5);
            xlabel('时间 (s)');
            ylabel('位移输出 (\mum)');
            title('作动器位移响应');
            grid on;
        end
    end
end