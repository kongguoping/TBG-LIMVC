function [Z, F, B,alpha, A_tensor,obj_values] = TBG_LIMVC(X, n, c, numAnchor, numNearestAnchor, params, beta, delta)
% F-MALT: Fast Multi-view Clustering via Anchor Label Transmit with Tensor Structure Constraint
% 输入:
%   X: 1 x V 细胞数组，每个元素为 n x d_v 的数据矩阵
%   c: 聚类数量
%   B: 1 x V 细胞数组，每个元素为 n x m 的双边图矩阵
%   Z_init: 1 x V 细胞数组，每个元素为 m x c 的初始anchor标签矩阵
%   params: 结构体，包含算法参数
%   beta: 不平衡正则化参数
% 输出:
%   Z: 1 x V 细胞数组，优化后的anchor标签矩阵
%   F: 1 x V 细胞数组，成员矩阵 F^{(v)} = B^{(v)} * Z^{(v)}
%   alpha: V x 1 向量，视图权重
%   A_tensor: n x V x c 张量，低秩结构张量
%   obj_values: 每次迭代的目标函数值
    nV = length(X);  % 视图数量
    sX = [n, nV, c]; % 张量尺寸
    weight_vector = ones(1, nV)'; 
    rho = 1e-4; rank_fun =4;
    % 设置默认参数
    if nargin < 5
        params = struct();
    end
    default_params = struct('max_iter', 100, 'tol', 1e-6, 'mu', 1e-5, ...
                           'lambda', 1, 'p', 0.1, 'mu_increase_rate', 1.2, ...
                           'mu_max', 10, 'plot_obj', true, 'use_parallel', true); % 添加并行选项
    params = set_default_params(params, default_params);
    
    % 初始化变量
    V = nV;
    m = 2^numAnchor;
    B = cell(1, nV);
    
    % ==================== 并行初始化 B 和锚点 ====================
    if params.use_parallel
        parfor v = 1:nV
            data_var = mean(var(X{v}));
            alpha_val = 2 / data_var;
            [~, locAnchor{v}] = hKM_EKM(X{v}', 1:n, numAnchor, 1, alpha_val);
            M = locAnchor{v}';
            [B{v}] = calculateZandF_EKM(X{v}, M, numNearestAnchor);
        end
    else
        for v = 1:nV
            data_var = mean(var(X{v}));
            alpha_val = 2 / data_var;
            [~, locAnchor{v}] = hKM_EKM(X{v}', 1:n, numAnchor, 1, alpha_val);
            M = locAnchor{v}';
            [B{v}] = calculateZandF_EKM(X{v}, M, numNearestAnchor);
        end
    end
    
    % 生成初始anchor标签矩阵 Z
    Z = cell(1, nV);
    % 初始化成员矩阵 F
    F = cell(1, V);
    
    % ==================== 并行初始化 Z 和 F ====================
    if params.use_parallel
        parfor v = 1:V
            Z{v} = initializeU(2^numAnchor, c);
            F{v} = B{v} * Z{v};
        end
    else
        for v = 1:V
            Z{v} = initializeU(2^numAnchor, c);
             % Z{v} = initializeUk_means(M, c);
            F{v} = B{v} * Z{v};
        end
    end
    
    % 构建张量 F_tensor (n x c x V)
    F_tensor_original = zeros(n, c, V);
    for v = 1:V
        F_tensor_original(:, :, v) = F{v};
    end
    
    % 重新排列张量维度为 n x V x c
    % 初始化alpha为均匀权重
    alpha = ones(V, 1) / V;
    
    % 初始化低秩张量 A_tensor
    A_tensor = zeros(n, V, c);
    
    % 存储目标函数值
    obj_values = zeros(params.max_iter, 1);
    
    for iter = 1:params.max_iter
        % 动态更新mu
        current_mu = min(params.mu * (params.mu_increase_rate^(iter-1)), params.mu_max);
        
        % ==================== Step 1: 并行更新所有视图的anchor标签矩阵 Z^v ====================
        if params.use_parallel
            parfor v = 1:V
                % 获取当前视图对应的A切片 (n x c)
                A_slice = squeeze(A_tensor(:, v, :));
                [Z{v}, ~] = update_U_multiview(B{v}, A_slice, Z{v}, alpha(v), beta, current_mu);
                F{v} = B{v} * Z{v};
                F_tensor_original(:, :, v) = F{v};
            end
        else
            for v = 1:V
                % 获取当前视图对应的A切片 (n x c)
                A_slice = squeeze(A_tensor(:, v, :));
                [Z{v}, ~] = update_U_multiview(B{v}, A_slice, Z{v}, alpha(v), beta, current_mu);
                F{v} = B{v} * Z{v};
                F_tensor_original(:, :, v) = F{v};
            end
        end
        
        F_tensor = permute(F_tensor_original, [1, 3, 2]);
        
        % Step 2: 更新低秩张量 A_tensor
        Fg = F_tensor(:); % 向量化
        
        %% ============================= Update J^k ==============================
        if rank_fun == 5  % TNN
            [j, objV] = wshrinkObj(Fg, weight_vector./rho, sX, 0, 3);
            A_tensor = reshape(j, sX);
        else     
            A_tensor = solve_G_fun(Fg, rho, sX, delta, rank_fun);
        end
        
        %% Step 3: 更新视图权重 alpha
        alpha = update_alpha(Z, B, V, c);
        
        % 计算目标函数值
        current_obj = compute_objective_multiview(Z, B, alpha, F_tensor, A_tensor, beta, current_mu, params.lambda, params.p);
        obj_values(iter) = current_obj;
        
        if mod(iter, 10) == 0 || iter == 1
            fprintf('迭代 %3d: 目标函数值 = %.6f, mu = %.2e\n', ...
                    iter, obj_values(iter), current_mu);
        end
        
        % 检查收敛
        if iter > 1 && abs(obj_values(iter) - obj_values(iter-1)) < params.tol
            fprintf('在迭代 %d 收敛\n', iter);
            obj_values = obj_values(1:iter);
            break;
        end
    end
    
    % 如果迭代完成但未提前收敛，截断obj_values
    if iter == params.max_iter
        obj_values = obj_values(1:iter);
        fprintf('达到最大迭代次数 %d\n', iter);
    end
end

% 辅助函数：设置默认参数
function params = set_default_params(params, default_params)
    field_names = fieldnames(default_params);
    for i = 1:length(field_names)
        if ~isfield(params, field_names{i})
            params.(field_names{i}) = default_params.(field_names{i});
        end
    end
end