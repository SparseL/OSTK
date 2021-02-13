linestyles={'k--','k-','b--','b-','r--','r-','m--','m-'};
for i = 1:1
    for j = 1:8
%         for k = 1:8
% plot(reshape(RMSEtrain(:,j,1),[1,8]),linestyles{j},'linewidth',1);hold on;
plot(reshape(RMSEtrain(:,1,j),[1,8]),linestyles{j},'linewidth',1);hold on;
% figure
% plot(reshape(RMSEtrain(i,:,1),[1,8]),'b-','linewidth',2)
%         end
    end
end
set(gca,'FontSize',12)  %是设置刻度字体大小
legend('\lambda_2=10^0','\lambda_2=10^{-1}','\lambda_2=10^{-2}','\lambda_2=10^{-3}','\lambda_2=10^{-4}',...
    '\lambda_2=10^{-5}','\lambda_2=10^{-6}','\lambda_2=10^{-7}','FontSize',12)
xlabel('\alpha','FontSize',14,'FontWeight','bold'); ylabel('RMSE','FontSize',14,'FontWeight','bold'); 
% axis tight;
set(gca,'XTickLabel',{'10^0','10^{-1}','10^{-2}','10^{-3}','10^{-4}','10^{-5}','10^{-6}','10^{-7}'});