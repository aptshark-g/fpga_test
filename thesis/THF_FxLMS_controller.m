% 弃用
function [y_thf, w, xf_buf, mu_eff] = THF_FxLMS_controller(x_ref, e, S_est, w, x_buf, xf_buf, mu_base, a, L, t_now, err_env, K_comp)
% 防御型变步长 THF-FxLMS（带输出增益补偿）
    persistent mu_max mu_min cold_start_duration
    if isempty(mu_max)
        mu_max = 5e-6;
        mu_min = 5e-8;
        cold_start_duration = 1.5;
    end
    
    if t_now < cold_start_duration
        mu = mu_max;
    else
        mu = mu_min + (mu_max - mu_min) * exp(-err_env / 2.0);
    end
    mu_eff = mu;
    
    y_lin = w' * x_buf;
    y_thf_raw = a * tanh(y_lin / a);
    y_thf = K_comp * y_thf_raw;   % 增益补偿，使控制量匹配次级路径衰减
    
    xf = sum(S_est .* x_buf(1:length(S_est)));
    xf_buf = [xf; xf_buf(1:end-1)];
    
    grad_factor = sech(y_lin / a)^2;
    w = w - mu * e * grad_factor * xf_buf;
    w = max(min(w, 2), -2);
end

function y = sech(x)
    y = 1 / cosh(x);
end