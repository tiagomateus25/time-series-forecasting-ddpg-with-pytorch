%% all data
name="traj";
idx=[1,2,3,4,5,6];
for i=4
    file=name+idx(i);
    load_f=load(file,"-mat");
    data=load_f.(file);
    data(:,[1,2,4,5,6])=[];
    writematrix(data,file+'_allData'+'.csv')
end

%% training and testing data
name="traj";
idx=[1,2,3,4,5,6];

% for i=1
%     file=name+idx(i);
%     load_f=load(file,"-mat");
%     data=load_f.(file);
%     data(:,[1,2,4,5,6,7])=[];
%     training_data=data(1:14915);
%     testing_data=data(14916:end);
%     writematrix(training_data,file+'_trainingData'+'.csv')
%     writematrix(testing_data,file+'_testingData'+'.csv')
% end
% 
% for i=2
%     file=name+idx(i);
%     load_f=load(file,"-mat");
%     data=load_f.(file);
%     data(:,[1,2,4,5,6,7])=[];
%     training_data=data(1:14788);
%     testing_data=data(14789:end);
%     writematrix(training_data,file+'_trainingData'+'.csv')
%     writematrix(testing_data,file+'_testingData'+'.csv')
% end
% 
% for i=3
%     file=name+idx(i);
%     load_f=load(file,"-mat");
%     data=load_f.(file);
%     data(:,[1,2,4,5,6,7])=[];
%     training_data=data(1:14779);
%     testing_data=data(14780:end);
%     writematrix(training_data,file+'_trainingData'+'.csv')
%     writematrix(testing_data,file+'_testingData'+'.csv')
% end

for i=4
    file=name+idx(i);
    load_f=load(file,"-mat");
    data=load_f.(file);
    data(:,[1,2,4,5,6,7])=[];
    training_data=data(1:14839);
    testing_data=data(14840:end);
    writematrix(training_data,file+'_trainingData'+'.csv')
    writematrix(testing_data,file+'_testingData'+'.csv')
end

% for i=5
%     file=name+idx(i);
%     load_f=load(file,"-mat");
%     data=load_f.(file);
%     data(:,[1,2,4,5,6,7])=[];
%     training_data=data(1:14924);
%     testing_data=data(14925:end);
%     writematrix(training_data,file+'_trainingData'+'.csv')
%     writematrix(testing_data,file+'_testingData'+'.csv')
% end

