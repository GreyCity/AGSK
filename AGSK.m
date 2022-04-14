%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluating the Performance of Adaptive Gaining-Sharing Knowledge Based
%Algorithm on CEC 2020 Benchmark Problems
%Authors: Ali W. Mohamed, Anas A. Hadi , Ali K. Mohamed, Noor H. Awad
%Modify/Note: GreyCity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;

format long;
Alg_Name='AGSK';  % 算法名称
n_problems=30;
ConvDisp=1;
Run_No=51;  % 计算轮数：30


for problem_size =[10 30 50 100]  % CEC2017的问题维度
	
    max_nfes = 10000 * problem_size;  % 最大操作数
    val_2_reach = 10^(-8);  % 最小误差
    max_region = 100.0;  % 上下边界100，-100
    min_region = -100.0;
    lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];  % size = 2*problem_size 第一行下边界，第二行上边界
    fhd=@cec17_func;  % 调用CEC2017的函数
    analysis= zeros(30,6);  % 用于存储30个问题的6项分析数据
    
    KF_pool = [0.1 1.0 0.5 1.0];  % 知识因子池(knowledge factor pool)：4种选择方案
    KR_pool = [0.2 0.1 0.9 0.9];  % 知识比率池(knowledge ratio pool)：4种选择方案
    
    
    for func = 1:n_problems  % CEC2017的30个问题

        optimum= func * 100.0;  % 真实全局最优解
        % Record the best results
        outcome = [];  % 记录误差
        fprintf('\n-------------------------------------------------------\n')
        fprintf('Function = %d, Dimension size = %d\n', func, problem_size)
        dim1=[];
        dim2=[];
		
        for run_id = 1 : Run_No

            rng(problem_size*func*run_id,'twister'); %To Check
            run_funcvals = [];  % 存储最优值
            bsf_error_val=[];  % 存储误差值
            % parameter settings for pop-size 初始种群大小NP
            pop_size = 100;  % 初始种群大小
            max_pop_size = pop_size;  % 种群最大值:初始种群大小
            min_pop_size = 12;  % 种群最小值:12
            
            % Initialize the main population 随机初始化种群：下边界+rand*(上边界-下边界)
            popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
            pop = popold; % the old population becomes the current population
			
            fitness = fhd(pop',func);  % 计算适应度值(消耗fes)
            fitness = fitness';
            
            nfes = 0;  % 适应度计数器
            bsf_fit_var = 1e+300;  % 最优值初始化为一个大数
            
            for i = 1 : pop_size  % 找到最小(优)值
                nfes = nfes + 1;
                if nfes > max_nfes; break; end
                if fitness(i) < bsf_fit_var
                    bsf_fit_var = fitness(i);  % 最优值
                end
                run_funcvals = [run_funcvals;bsf_fit_var];  % 记录最优值
            end
            
           
            % POSSIBLE VALUES FOR KNOWLEDGE RATE K 
            K=[];
            KF=[];
            KR=[];
            Kind=rand(pop_size, 1);  % 计算NP个个体的K值，0.5的概率在取值在(0,1)的随机小数，0.5的概率在取值在[1,20]的随机整数
            K(Kind<0.5,:)= rand(sum(Kind<0.5), 1);
            K(Kind>=0.5,:)=ceil(20 * rand(sum(Kind>=0.5), 1));  % K值越大，初级知识越短，高级知识越长
       
            g=0;  % 种群迭代次数初始化为0
            % main loop
            
            KW_ind=[];
            All_Imp=zeros(1,4);

            while nfes < max_nfes
                g=g+1;
             
                 if  (nfes < 0.1*max_nfes)% intial probability values 
                    KW_ind=[0.85 0.05 0.05 0.05];  % 以0.85的概率选方案1,0.05选方案2，0.05选方案3，0.05选方案4
                    K_rand_ind=rand(pop_size, 1);
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:3))&K_rand_ind<=sum(KW_ind(1:4)))=4;
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:2))&K_rand_ind<=sum(KW_ind(1:3)))=3;
                    K_rand_ind(K_rand_ind>KW_ind(1)&K_rand_ind<=sum(KW_ind(1:2)))=2;
                    K_rand_ind(K_rand_ind>0&K_rand_ind<=KW_ind(1))=1;
                    KF=KF_pool(K_rand_ind)';
                    KR=KR_pool(K_rand_ind)';
                 else % updaing probability values  调整选方案的概率
                    KW_ind=0.95*KW_ind+0.05*All_Imp;
                    KW_ind=KW_ind./sum(KW_ind);
                    K_rand_ind=rand(pop_size, 1);
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:3))&K_rand_ind<=sum(KW_ind(1:4)))=4;
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:2))&K_rand_ind<=sum(KW_ind(1:3)))=3;
                    K_rand_ind(K_rand_ind>KW_ind(1)&K_rand_ind<=sum(KW_ind(1:2)))=2;
                    K_rand_ind(K_rand_ind>0&K_rand_ind<=KW_ind(1))=1;
                    KF=KF_pool(K_rand_ind)';
                    KR=KR_pool(K_rand_ind)';
                    
                end
                
                % Junior and Senior Gaining-Sharing phases 
                D_Gained_Shared_Junior=ceil((problem_size)*(1-nfes / max_nfes).^K);  % 初级知识长度
                D_Gained_Shared_Senior=problem_size-D_Gained_Shared_Junior;  % 高级知识长度
                pop = popold; % the old population becomes the current population
                
                [valBest, indBest] = sort(fitness, 'ascend');  % 适应度升序排列
                [Rg1, Rg2, Rg3] = Gained_Shared_Junior_R1R2R3(indBest);  % Rg1:左邻居 Rg2:右邻居 Rg3:随机个体
                
                [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest);  % R1:Best People R2:Better People R3:Worst People
                R01=1:pop_size;
                Gained_Shared_Junior=zeros(pop_size, problem_size);
                ind1=fitness(R01)>fitness(Rg3);  % 计算初级交叉
                
                if(sum(ind1)>0)
                    Gained_Shared_Junior (ind1,:)= pop(ind1,:) + KF(ind1, ones(1,problem_size)).* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(Rg3(ind1), :)-pop(ind1,:)) ;
                end
                ind1=~ind1;
                if(sum(ind1)>0)
                    Gained_Shared_Junior(ind1,:) = pop(ind1,:) + KF(ind1, ones(1,problem_size)) .* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(ind1,:)-pop(Rg3(ind1), :)) ;
                end
                R0=1:pop_size;
                Gained_Shared_Senior=zeros(pop_size, problem_size);
                ind=fitness(R0)>fitness(R2);  % 计算高级交叉
                if(sum(ind)>0)
                    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF(ind, ones(1,problem_size)) .* (pop(R1(ind),:) - pop(ind,:) + pop(R2(ind),:) - pop(R3(ind), :)) ;
                end
                ind=~ind;
                if(sum(ind)>0)
                    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF(ind, ones(1,problem_size)) .* (pop(R1(ind),:) - pop(R2(ind),:) + pop(ind,:) - pop(R3(ind), :)) ;
                end
                Gained_Shared_Junior = boundConstraint(Gained_Shared_Junior, pop, lu);  % 边界处理
                Gained_Shared_Senior = boundConstraint(Gained_Shared_Senior, pop, lu);
                
                % 初级知识位置 = 全区域(NP*D)内以初级知识占比的概率(D_junior/D)随机选择
                D_Gained_Shared_Junior_mask=rand(pop_size, problem_size)<=(D_Gained_Shared_Junior(:, ones(1, problem_size))./problem_size); % mask is used to indicate which elements of will be gained
                D_Gained_Shared_Senior_mask=~D_Gained_Shared_Junior_mask;  % 初级知识位置和高级知识位置
                
                D_Gained_Shared_Junior_rand_mask=rand(pop_size, problem_size)<=KR(:,ones(1, problem_size));
                D_Gained_Shared_Junior_mask=and(D_Gained_Shared_Junior_mask,D_Gained_Shared_Junior_rand_mask);  % 初级交叉的位置
                
                D_Gained_Shared_Senior_rand_mask=rand(pop_size, problem_size)<=KR(:,ones(1, problem_size));
                D_Gained_Shared_Senior_mask=and(D_Gained_Shared_Senior_mask,D_Gained_Shared_Senior_rand_mask);  % 高级交叉的位置
                ui=pop;
                
                ui(D_Gained_Shared_Junior_mask) = Gained_Shared_Junior(D_Gained_Shared_Junior_mask);  % 进行初级交叉
                ui(D_Gained_Shared_Senior_mask) = Gained_Shared_Senior(D_Gained_Shared_Senior_mask);  % 进行高级交叉
                
                children_fitness = fhd(ui', func);
                children_fitness = children_fitness';  % 计算交叉后的适应度(消耗fes)
                % Updating individuals
                for i = 1 : pop_size
                    nfes = nfes + 1;
                    if nfes > max_nfes; break; end
                    if children_fitness(i) < bsf_fit_var
                        bsf_fit_var = children_fitness(i);  % 更新最优值
                        bsf_solution = ui(i, :);  % 更新最优解
                    end
                    run_funcvals = [run_funcvals;bsf_fit_var];
                end
                % Calculate the improvemnt of each settings
                dif = abs(fitness - children_fitness);
                Child_is_better_index = (fitness > children_fitness);  % 交叉后更好的个体的位置
                dif_val = dif(Child_is_better_index == 1);  % 提高 size = 交叉后更好的数量
                All_Imp=zeros(1,4);% (1,4) delete for 4 cases
                for i=1:4
                    if(sum(and(Child_is_better_index,K_rand_ind==i))>0)  % 如果方案i带来的提高的个体数量大于零
                        All_Imp(i)=sum(dif(and(Child_is_better_index,K_rand_ind==i)));  % 方案i带来的总提高
                    else
                        All_Imp(i)=0;
                    end
                end
                
                if(sum(All_Imp)~=0)
                    All_Imp=All_Imp./sum(All_Imp);  % 归一化
                    [temp_imp,Imp_Ind] = sort(All_Imp);  % temp_imp是升序排列 Imp_Ind是下标
                    for imp_i=1:length(All_Imp)-1
                        All_Imp(Imp_Ind(imp_i))=max(All_Imp(Imp_Ind(imp_i)),0.05);  % 优化效果应该不低于0.05
                    end
                    All_Imp(Imp_Ind(end))=1-sum(All_Imp(Imp_Ind(1:end-1)));
                else
                    Imp_Ind=1:length(All_Imp);
                    All_Imp(:)=1/length(All_Imp);  % 没有提高则等概分布
                end
                [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);
                
                popold = pop;
                popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);  % 更新种群
                % for resizing the population size
               
                plan_pop_size = round((min_pop_size - max_pop_size)* ((nfes / max_nfes).^((1-nfes / max_nfes)))  + max_pop_size);  % max_pop_size = NP 种群个体数量线性减少
               
                if pop_size > plan_pop_size
                    reduction_ind_num = pop_size - plan_pop_size;
                    if (pop_size - reduction_ind_num <  min_pop_size)
                        reduction_ind_num = pop_size - min_pop_size;
                    end
                    
                    pop_size = pop_size - reduction_ind_num;  % 计算新的种群个体数量
                    for r = 1 : reduction_ind_num
                        [valBest,indBest] = sort(fitness, 'ascend');
                        worst_ind = indBest(end);
                        popold(worst_ind,:) = [];  % 把最差的那部分清空
                        pop(worst_ind,:) = [];
                        fitness(worst_ind,:) = [];
                        K(worst_ind,:)=[];
                    end
                end
                
            end  % end while loop

            bsf_error_val = bsf_fit_var - optimum;
            if bsf_error_val < val_2_reach
                bsf_error_val = 0;
            end
			
            fprintf('%d th run, best-so-far error value = %1.8e\n', run_id , bsf_error_val)  % 每轮报告
            outcome = [outcome bsf_error_val];  % 记录误差
            % Save Convergence Figures
            if (ConvDisp)
                run_funcvals=run_funcvals-optimum;
                run_funcvals=run_funcvals';
                dim1(run_id,:)=1:length(run_funcvals);
                dim2(run_id,:)=log10(run_funcvals);
            end
            
        end  % end 1 run
        
        %Save results
        analysis(func,1)=min(outcome);
        analysis(func,2)=median(outcome);
        analysis(func,3)=max(outcome);
        analysis(func,4)=mean(outcome);
        analysis(func,5)=std(outcome);
        median_figure=find(outcome== median(outcome));
        analysis(func,6)=median_figure(1);
		
        file_name=sprintf('Results\\%s_CEC2017_Problem#%s_problem_size#%s',Alg_Name,int2str(func),int2str(problem_size));  % 保存误差文件
        save(file_name,'outcome');
        
        fprintf('min:%e\n',min(outcome));
        fprintf('median:%e\n',median(outcome));
        fprintf('mean:%e\n',mean(outcome));
        fprintf('max:%e\n',max(outcome));
        fprintf('std:%e\n',std(outcome));
        dim11=dim1(median_figure,:);
        dim22=dim2(median_figure,:);
        file_name=sprintf('Figures\\Figure_Problem#%s_Run#%s',int2str(func),int2str(run_id));
        save(file_name,'dim11','dim22');  % 保存中值
    end  % end 1 function run
	
    file_name=sprintf('Results\\analysis_%s_CEC2017_problem_size#%s',Alg_Name,int2str(problem_size));  % 保存文件为analysis_GSK_CEC2017_problem_size#10.mat
    save(file_name,'analysis');  % 保存数据分析
	
end  % end all function runs in all dimensions


