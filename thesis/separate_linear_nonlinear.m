function [s_lin, s_nl, state] = separate_linear_nonlinear(s, state, varargin)
% 基于自适应线性预测器将信号分离为线性可预测部分和非线性残差部分
% 输入：
%   s     - 当前时刻的标量输入
%   state - 结构体，包含预测器状态（w, buffer, mu, order）
%   varargin - 可选参数：'mu', 'order'
% 输出：
%   s_lin - 线性预测部分
%   s_nl  - 非线性残差部分
%   state - 更新后的状态

    % 默认参数
    if isempty(state)
        state.order = 128;          % 预测器阶数
        state.mu = 0.01;           % 自适应步长
        state.w = zeros(state.order, 1);
        state.buffer = zeros(state.order, 1);
    end
    
    % 允许在调用时覆盖参数
    for i = 1:2:length(varargin)
        switch varargin{i}
            case 'mu', state.mu = varargin{i+1};
            case 'order', state.order = varargin{i+1};
        end
    end
    
    % 更新缓冲区
    state.buffer = [s; state.buffer(1:end-1)];
    % 线性预测输出
    s_lin = state.w' * state.buffer;
    % 非线性残差
    s_nl = s - s_lin;
    % 更新预测器系数（LMS）
    error = s - s_lin;
    state.w = state.w + state.mu * error * state.buffer;
end