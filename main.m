clear; clc;
folder_now = pwd;
addpath([folder_now, '\funs']);
addpath([folder_now, '\dataset']);

dataname = ["MSRC"];

%% ==================== Load Dataset and Normalization ====================
for it_name = 1:length(dataname)

    load(strcat('dataset/', dataname(it_name), '.mat'));
    % X = {f_list_0, f_list_1};
    gt =Y;
    n = length(gt);
    cls_num = length(unique(gt));

    % 归一化
    nV = length(X);
    for v = 1:nV
        X{v} = NormalizeData(X{v});
    end
    for i = 1:length(X)
        if size(X{i},1) == length(gt)
            X{i} = X{i};
        else
            X{i} = double(X{i}');
        end
    end

    %% 参数设置
    params = struct();
    params.max_iter = 100;
    params.tol = 1e-6;
    params.mu = 1e-5;
    params.p = 0.8;
    params.mu_increase_rate = 1.2;
    params.mu_max = 10;

    result = [];
    ii = 0;
    max_val = 0;

    %% 搜索范围
    numNearestAnchor_range = 5:1:7;
    beta_range = -4:1:4;       % beta = 10^(i_beta)
    delta_range = -2:1:1;      % delta = 10^(i_delta)
    anchor_range = 5:1:9;      % number of anchors
    rho_range = -4:1:0;        % rho = 10^(i_rho)

    total_iterations = length(numNearestAnchor_range) * ...
                       length(beta_range) * ...
                       length(delta_range) * ...
                       length(anchor_range) * ...
                       length(rho_range);

    fprintf('开始实验 %s，共 %d 种参数组合\n', dataname(it_name), total_iterations);

    current_iteration = 0;

    %% ============================ Grid Search ============================
    for i_numNearestAnchor = numNearestAnchor_range
        for i_beta = beta_range
            for i_delta = delta_range
                for i_anchor = anchor_range
                    for i_rho = rho_range

                        numNearestAnchor = i_numNearestAnchor;
                        beta = 10^(i_beta);
                        delta = 10^(i_delta);
                        anc = i_anchor;
                        rho = 10^(i_rho);

                        % 写入 params
                        params.rho = rho;

                        ii = ii + 1;
                        current_iteration = current_iteration + 1;
                        tic;

                        progress_percent = round((current_iteration / total_iterations) * 100, 1);
                        fprintf(['进度 %d/%d (%.1f%%) - 参数: numA=%d, beta=10^{%d}, ' ...
                                 'delta=10^{%d}, anchor=%d, rho=10^{%d}\n'], ...
                                 current_iteration, total_iterations, progress_percent, ...
                                 numNearestAnchor, i_beta, i_delta, anc, i_rho);

                        %% 调用算法
                        [Z_opt, F_opt,B, alpha_opt, A_opt,obj_values] = ...
                            TBG_LIMVC(X, n, cls_num, anc, numNearestAnchor, params, beta, delta);

                        unified_labels = get_unified_labels_from_A(A_opt);

                        time = toc;

                        [result(ii,:), ~] = MyClusteringMeasure(gt, unified_labels);

                        if result(ii,1) > max_val
                            max_val = result(ii,1);
                            record = [numNearestAnchor, beta, delta, anc, rho, time];
                            record_result = result(ii,:);
                        end
                    end
                end
            end
        end
    end

    disp('最佳结果：'); 
    record_result

    save(['.\result_ours\result_ours_' dataname(it_name)], ...
         'result', 'record', 'max_val', 'record_result')
end
