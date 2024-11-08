
clear all
clc
close all

OriginalData=[   0.3245   -0.3728    0.2028   -0.2772    0.1435
                            0.3388    0.2255   -0.0012   -0.8165   -0.0056
                           -0.0375    0.4158    0.0965   -0.5582   -0.2763
                            0.3505    0.1603   -0.2610   -0.4725    0.2020
                            0.4649    0.0700    0.0078   -0.8353    0.1016
                            0.6434   -0.4726    0.0177   -0.4610    0.4111
                            0.5504   -0.5524   -0.2517    0.0782    0.6005
                            0.4228   -0.4298   -0.0617   -0.0761    0.3765
                            0.1559   -0.0168    0.1519   -0.3926   -0.0457
                            0.3037    0.1860    0.0574   -0.7755   -0.0362];

[new_col_order , CoveredR2 , OrganizedData]=SPCA(OriginalData)




function [new_col_order , CoveredR2 , OrganizedData]=SPCA(data)

%%% receive the block of data and sort it based on its columns covered
%%% variance of the data, and also plot the results showing with how many
%%% first columns of this sorted data how much variance of the data is
%%% covered

            Num_directions=size(data,2);
            CoveredR2=zeros(1,Num_directions);
            original_idx=1:size(data,2);
            new_col_order=zeros(1,Num_directions); 

            data_scaled=(data-mean(data))./(std(data)+1e-16);
            data_scaled_Original=data_scaled;

            for i=1:Num_directions-1

                [P,~]=pca(data_scaled);

                P=P(:,1);

                [~, best_col]=max(abs(P));                

                Y_best=data_scaled(:,best_col);

                data_scaled(:,best_col)=[];

                p_new=(data_scaled'*Y_best)/(Y_best'*Y_best);

                E_new=data_scaled-(Y_best*p_new');

                CoveredR2(i)=1-(sum(var(E_new))/sum(var(data_scaled_Original)))-sum(CoveredR2(1:i-1));

                data_scaled=E_new;

                new_col_order(i)=original_idx(best_col);

                original_idx(best_col)=[];
         
            end

            new_col_order(end)=setdiff(1:Num_directions,new_col_order);

            CoveredR2(end)=1-sum(CoveredR2);

            OrganizedData=data(:,new_col_order);
        
            spcsplotter(CoveredR2,new_col_order)

            

end


function []=spcsplotter(coveredR2,ordered_col)

    Num_directions=numel(coveredR2);
    figure

        subplot(2,1,1)
            bar(coveredR2*100)
            hold on
            plot([0 Num_directions+1],90*[1 1],'k--',LineWidth=2)

            for i=1:Num_directions
            xtich_str{i}=['col ', num2str(ordered_col(i))];
            end
            
            xticklabels(xtich_str)
            xticks(1:Num_directions)
            xlabel('Sorted Original Column Numbers')
            ylabel('Covered Variance (%)')
            title('Variance Covered for Individual Columns(Sorted)')
            set(gca, 'LineWidth', 2, 'FontSize', 15);
 

        
        

        subplot(2,1,2)
       
        bar(cumsum(coveredR2)*100)
            hold on
            plot([0 Num_directions+1],90*[1 1],'k--',LineWidth=2)

            for i=1:Num_directions
            xtich_str{i}=[num2str(i),' columns '];
            end
            
            legend('','90 % Variance Covered')
            xticklabels(xtich_str)
            xticks(1:Num_directions)
            xlabel('Cumulative Columns Selected')
            ylabel('Covered Variance (%)')
            title('Variance Covered for n Columns')
            set(gca, 'LineWidth', 2, 'FontSize', 15);
      
end




