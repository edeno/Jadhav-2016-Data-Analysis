function [rippleI, ...
    ripple_index, ...
    position_time_stamps, ...
    position_time_stamps_binned, ...
    vecLF, ...
    traj_Ind, ...
    time, ...
    stateV, ...
    stateV_delta, ...
    stateM_I1_normalized_gaussian, ...
    stateM_Indicator0_normalized_gaussian, ...
    Lint_Indicator_outbound, ...
    Lint_Indicator_inbound, ...
    tet_ind, ...
    tet_sum, ...
    markAll_t1, ...
    markAll_t2, ...
    markAll_t4, ...
    markAll_t5, ...
    markAll_t7, ...
    markAll_t10, ...
    markAll_t11, ...
    markAll_t12, ...
    markAll_t13, ...
    markAll_t14, ...
    markAll_t17, ...
    markAll_t18, ...
    markAll_t19, ...
    markAll_t20, ...
    markAll_t22, ...
    markAll_t23, ...
    markAll_t27, ...
    markAll_t29, ...
    procInd1_t1_Ia_out, ...
    procInd1_t2_Ia_out, ...
    procInd1_t4_Ia_out, ...
    procInd1_t5_Ia_out, ...
    procInd1_t7_Ia_out, ...
    procInd1_t10_Ia_out, ...
    procInd1_t11_Ia_out, ...
    procInd1_t12_Ia_out, ...
    procInd1_t13_Ia_out, ...
    procInd1_t14_Ia_out, ...
    procInd1_t17_Ia_out, ...
    procInd1_t18_Ia_out, ...
    procInd1_t19_Ia_out, ...
    procInd1_t20_Ia_out, ...
    procInd1_t22_Ia_out, ...
    procInd1_t23_Ia_out, ...
    procInd1_t27_Ia_out, ...
    procInd1_t29_Ia_out, ...
    procInd1_t1_I_out, ...
    procInd1_t2_I_out, ...
    procInd1_t4_I_out, ...
    procInd1_t5_I_out, ...
    procInd1_t7_I_out, ...
    procInd1_t10_I_out, ...
    procInd1_t11_I_out, ...
    procInd1_t12_I_out, ...
    procInd1_t13_I_out, ...
    procInd1_t14_I_out, ...
    procInd1_t17_I_out, ...
    procInd1_t18_I_out, ...
    procInd1_t19_I_out, ...
    procInd1_t20_I_out, ...
    procInd1_t22_I_out, ...
    procInd1_t23_I_out, ...
    procInd1_t27_I_out, ...
    procInd1_t29_I_out, ...
    smker, ...
    Xnum_t1_I_out, ...
    Xnum_t2_I_out, ...
    Xnum_t4_I_out, ...
    Xnum_t5_I_out, ...
    Xnum_t7_I_out, ...
    Xnum_t10_I_out, ...
    Xnum_t11_I_out, ...
    Xnum_t12_I_out, ...
    Xnum_t13_I_out, ...
    Xnum_t14_I_out, ...
    Xnum_t17_I_out, ...
    Xnum_t18_I_out, ...
    Xnum_t19_I_out, ...
    Xnum_t20_I_out, ...
    Xnum_t22_I_out, ...
    Xnum_t23_I_out, ...
    Xnum_t27_I_out, ...
    Xnum_t29_I_out, ...
    occ_Indicator_outbound, ...
    Lint_t1_I_out, ...
    Lint_t2_I_out, ...
    Lint_t4_I_out, ...
    Lint_t5_I_out, ...
    Lint_t7_I_out, ...
    Lint_t10_I_out, ...
    Lint_t11_I_out, ...
    Lint_t12_I_out, ...
    Lint_t13_I_out, ...
    Lint_t14_I_out, ...
    Lint_t17_I_out, ...
    Lint_t18_I_out, ...
    Lint_t19_I_out, ...
    Lint_t20_I_out, ...
    Lint_t22_I_out, ...
    Lint_t23_I_out, ...
    Lint_t27_I_out, ...
    Lint_t29_I_out, ...
    procInd1_t1_Ia_in, ...
    procInd1_t2_Ia_in, ...
    procInd1_t4_Ia_in, ...
    procInd1_t5_Ia_in, ...
    procInd1_t7_Ia_in, ...
    procInd1_t10_Ia_in, ...
    procInd1_t11_Ia_in, ...
    procInd1_t12_Ia_in, ...
    procInd1_t13_Ia_in, ...
    procInd1_t14_Ia_in, ...
    procInd1_t17_Ia_in, ...
    procInd1_t18_Ia_in, ...
    procInd1_t19_Ia_in, ...
    procInd1_t20_Ia_in, ...
    procInd1_t22_Ia_in, ...
    procInd1_t23_Ia_in, ...
    procInd1_t27_Ia_in, ...
    procInd1_t29_Ia_in, ...
    procInd1_t1_I_in, ...
    procInd1_t2_I_in, ...
    procInd1_t4_I_in, ...
    procInd1_t5_I_in, ...
    procInd1_t7_I_in, ...
    procInd1_t10_I_in, ...
    procInd1_t11_I_in, ...
    procInd1_t12_I_in, ...
    procInd1_t13_I_in, ...
    procInd1_t14_I_in, ...
    procInd1_t17_I_in, ...
    procInd1_t18_I_in, ...
    procInd1_t19_I_in, ...
    procInd1_t20_I_in, ...
    procInd1_t22_I_in, ...
    procInd1_t23_I_in, ...
    procInd1_t27_I_in, ...
    procInd1_t29_I_in, ...
    Xnum_t1_I_in, ...
    Xnum_t2_I_in, ...
    Xnum_t4_I_in, ...
    Xnum_t5_I_in, ...
    Xnum_t7_I_in, ...
    Xnum_t10_I_in, ...
    Xnum_t11_I_in, ...
    Xnum_t12_I_in, ...
    Xnum_t13_I_in, ...
    Xnum_t14_I_in, ...
    Xnum_t17_I_in, ...
    Xnum_t18_I_in, ...
    Xnum_t19_I_in, ...
    Xnum_t20_I_in, ...
    Xnum_t22_I_in, ...
    Xnum_t23_I_in, ...
    Xnum_t27_I_in, ...
    Xnum_t29_I_in, ...
    occ_Indicator_inbound, ...
    Lint_t1_I_in, ...
    Lint_t2_I_in, ...
    Lint_t4_I_in, ...
    Lint_t5_I_in, ...
    Lint_t7_I_in, ...
    Lint_t10_I_in, ...
    Lint_t11_I_in, ...
    Lint_t12_I_in, ...
    Lint_t13_I_in, ...
    Lint_t14_I_in, ...
    Lint_t17_I_in, ...
    Lint_t18_I_in, ...
    Lint_t19_I_in, ...
    Lint_t20_I_in, ...
    Lint_t22_I_in, ...
    Lint_t23_I_in, ...
    Lint_t27_I_in, ...
    Lint_t29_I_in ...
    ] = encode_state(animal, day, linpos, pos, trajencode, ripplescons, spikes, tetrode_index, neuron_index, tetrode_number)
%% use Loren's linearization
time=linpos.statematrix.time;
linear_distance=linpos.statematrix.lindist;
vecLF(:,1)=time;vecLF(:,2)=linear_distance;
%figure;plot(time,linear_distance,'.');

position_time_stamps=pos.data(:,1); %time stamps for animal's trajectory
position_time_stamps_binned=round(position_time_stamps(1)*1000):1:round(position_time_stamps(end)*1000); %binning time stamps at 1 ms
linear_distance_bins = 61;
stateV=linspace(min(linear_distance),max(linear_distance), linear_distance_bins);
stateV_delta=stateV(2)-stateV(1);


%% calculate emipirical movement transition matrix, then Gaussian smoothed
[~,state_bin]=histc(linear_distance,stateV);
state_disM=[state_bin(1:end-1) state_bin(2:end)];
stateV_length=size(stateV,2);
%by column=departuring
for stateV_ind=1:stateV_length
    sp0=state_disM(state_disM(:,1)==stateV_ind,2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
    if ~isempty(sp0)
        stateM(:,stateV_ind)=histc(sp0,linspace(1,stateV_length,stateV_length))./size(sp0,1);
    else
        stateM(:,stateV_ind)=zeros(1,stateV_length);
    end
end
gaussian=@(sigma, x, y) exp(-(x.^2+y.^2)/2/sigma^2); %gaussian

%%
ind_Indicator_outbound=find(trajencode.trajstate==1 | trajencode.trajstate==3);
ind_Indicator_inbound=find(trajencode.trajstate==2 | trajencode.trajstate==4);
%figure;plot(vecLF(ind_I_out,1),vecLF(ind_I_out,2),'r.',vecLF(ind_I_in,1),vecLF(ind_I_in,2),'b.');

%% empirical movement transition matrix conditioned on I=1(outbound) and I=0 (inbound)
stateV_length=length(stateV);
stateM_Indicator_outbound=zeros(stateV_length);
vecLF_seg=vecLF(ind_Indicator_outbound,:);
[~,state_bin]=histc(vecLF_seg(:,2),stateV);
state_disM=[state_bin(1:end-1) state_bin(2:end)];
stateM_seg=zeros(stateV_length);
for stateV_ind=1:stateV_length
    sp0=state_disM(state_disM(:,1)==stateV_ind,2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
    if isempty(sp0)==0
        stateM_seg(:,stateV_ind)=histc(sp0,linspace(1,stateV_length,stateV_length))./size(sp0,1);
    elseif isempty(sp0)==1
        stateM_seg(:,stateV_ind)=zeros(1,stateV_length);
    end
end
stateM_Indicator_outbound=stateM_Indicator_outbound+stateM_seg;
%%%if too many zeros:
for i=1:stateV_length
    if sum(stateM_Indicator_outbound(:,i))==0
        stateM_Indicator_outbound(:,i)=1/stateV_length;
    elseif sum(stateM_Indicator_outbound(:,i))~=0
        stateM_Indicator_outbound(:,i)=stateM_Indicator_outbound(:,i)./sum(stateM_Indicator_outbound(:,i));
    end
end
%stateM_I1=stateM_I1*diag(1./sum(stateM_I1,1));
[dx,dy]=meshgrid([-1:1]);
sigma=0.5;
normalizing_weight=gaussian(sigma,dx,dy)/sum(sum(gaussian(sigma,dx,dy))); %normalizing weights
stateM_gaussian_smoothed=conv2(stateM_Indicator_outbound,normalizing_weight,'same'); %gaussian smoothed
stateM_I1_normalized_gaussian=stateM_gaussian_smoothed*diag(1./sum(stateM_gaussian_smoothed,1)); %normalized to confine probability to 1


stateM_Indicator_inbound=zeros(stateV_length);
vecLF_seg=vecLF(ind_Indicator_inbound,:);
[~,  state_bin]=histc(vecLF_seg(:,2),stateV);
state_disM=[state_bin(1:end-1) state_bin(2:end)];
stateM_seg=zeros(stateV_length);
for stateV_ind=1:stateV_length
    sp0=state_disM(state_disM(:,1)==stateV_ind,2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
    if isempty(sp0)==0
        stateM_seg(:,stateV_ind)=histc(sp0,linspace(1,stateV_length,stateV_length))./size(sp0,1);
    elseif isempty(sp0)==1
        stateM_seg(:,stateV_ind)=zeros(1,stateV_length);
    end
end
stateM_Indicator_inbound=stateM_Indicator_inbound+stateM_seg;
%%if too many zeros
for i=1:stateV_length
    if sum(stateM_Indicator_inbound(:,i))==0
        stateM_Indicator_inbound(:,i)=1/stateV_length;
    elseif sum(stateM_Indicator_inbound(:,i))~=0
        stateM_Indicator_inbound(:,i)=stateM_Indicator_inbound(:,i)./sum(stateM_Indicator_inbound(:,i));
    end
end
%stateM_I0=stateM_I0*diag(1./sum(stateM_I0,1));

[dx,dy]=meshgrid([-1:1]);
sigma=0.5;
normalizing_weight=gaussian(sigma,dx,dy)/sum(sum(gaussian(sigma,dx,dy))); %normalizing weights
stateM_gaussian_smoothed=conv2(stateM_Indicator_inbound,normalizing_weight,'same'); %gaussian smoothed
stateM_Indicator0_normalized_gaussian=stateM_gaussian_smoothed*diag(1./sum(stateM_gaussian_smoothed,1)); %normalized to confine probability to 1


%% calculate ripple starting and end times
ripple_start_time=ripplescons{1}.starttime;
ripple_end_time=ripplescons{1}.endtime;
traj_Ind=find(ripplescons{1}.maxthresh>4);
ripple_start_time=ripple_start_time(traj_Ind);
ripple_end_time=ripple_end_time(traj_Ind);
ripple_index=[round(ripple_start_time*1000)-position_time_stamps_binned(1)-1, round(ripple_end_time*1000)-position_time_stamps_binned(1)-1]; %index for ripple segments

clear sptrain2_list;
for neuron_ind=1:size(tetrode_index,2)
    spike_times=spikes{tetrode_index(neuron_ind)}{neuron_index(neuron_ind)}.data(:,1); %spiking times for tetrode j, cell i
    xi=round(spike_times*1000); %binning spiking times at 1 ms
    [sptrain2,~]=ismember(position_time_stamps_binned,xi); %sptrain2: spike train binned at 1 ms instead of 33.4ms (sptrain0)
    sptrain2_list{neuron_ind}=sptrain2;
end

clear spike_r_all;
for k=1:size(ripple_index,1)
    spike_r=[];
    for neuron_ind=1:size(tetrode_index,2)
        sptrain2=sptrain2_list{neuron_ind};
        spike_r=[spike_r;sptrain2(ripple_index(k,1):ripple_index(k,2))];
    end
    spike_r_all{k}=spike_r;
end

clear sumR;
for k=1:size(ripple_index,1)
    spike_r=spike_r_all{k};
    sumR(k)=sum(spike_r(:));
end
rippleI=find(sumR>0);

%% prepare kernel density model
linear_position_time = linpos.statematrix.time;

poslin = vecLF(:,2);
xs = min(poslin):stateV_delta:max(poslin);
dt = linear_position_time(2)-linear_position_time(1);
xtrain = poslin';

sxker = stateV_delta;
mdel = 20;
smker = mdel;
num_time_points_position = size(linear_position_time, 1);

%% encode the kernel density model per tetrode

num_tetrodes = length(tetrode_number);

markAll = cell(num_tetrodes, 1);
time0 = cell(num_tetrodes, 1);
mark0 = cell(num_tetrodes, 1);
procInd1 = cell(num_tetrodes, 1);

for tetrode_ind = 1:num_tetrodes,
    [markAll{tetrode_ind}, time0{tetrode_ind}, mark0{tetrode_ind}, ...
        procInd1{tetrode_ind}] = kernel_density_model(animal, day, tetrode_number(tetrode_ind), ...
        linear_position_time, mdel, sxker, xs, xtrain, num_time_points_position, dt);
end

mark0 = cat(1, mark0{:});
procInd1 = cat(1, procInd1{:});
%% bookkeeping code: which spike comes which tetrode
[time, timeInd] = sort(cat(1, time0{:}));
mark0 = mark0(timeInd, :);
procInd1=procInd1(timeInd, :);

tet_ind = false(length(time), num_tetrodes);

for tetrode_ind = 1:num_tetrodes,
    tet_ind(:, tetrode_ind) = ismember(time, time0{tetrode_ind});
end

tet_sum=tet_ind .* cumsum(tet_ind,1); %row: time point; column: index of spike per tetrode

%% caculate captial LAMBDA (when there is no spike on any tetrode)
ms=min(mark0(:)):mdel:max(mark0(:));
occ=normpdf(xs' * ones(1,num_time_points_position), ones(length(xs),1)*xtrain,sxker) * ones(num_time_points_position, length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum = normpdf(xs' * ones(1, length(time)), ones(length(xs),1) * xtrain(procInd1), sxker);
%Xnum: Gaussian kernel estimators for position
Lint=sum(Xnum,2)./occ(:,1)./dt; %integral
Lint=Lint./sum(Lint);
%Lint: conditional intensity function for the unmarked case



%% captial LAMBDA conditioned on I=1 and I=0
procInd1_Indicator_outbound=procInd1(ismember(procInd1,ind_Indicator_outbound));
occ_Indicator_outbound=normpdf(xs'*ones(1,length(ind_Indicator_outbound)),ones(length(xs),1)*xtrain(ind_Indicator_outbound),sxker)*ones(length(ind_Indicator_outbound),length(ms));
Xnum_Indicator_outbound=normpdf(xs'*ones(1,length(xtrain(procInd1_Indicator_outbound))),ones(length(xs),1)*xtrain(procInd1_Indicator_outbound),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_Indicator_outbound=sum(Xnum_Indicator_outbound,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_Indicator_outbound=Lint_Indicator_outbound./sum(Lint_Indicator_outbound);


procInd1_I_inbound=procInd1(ismember(procInd1,ind_Indicator_inbound));
occ_Indicator_inbound=normpdf(xs'*ones(1,length(ind_Indicator_inbound)),ones(length(xs),1)*xtrain(ind_Indicator_inbound),sxker)*ones(length(ind_Indicator_inbound),length(ms));
Xnum_Indicator_inbound=normpdf(xs'*ones(1,length(xtrain(procInd1_I_inbound))),ones(length(xs),1)*xtrain(procInd1_I_inbound),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_Indicator_inbound=sum(Xnum_Indicator_inbound,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_Indicator_inbound=Lint_Indicator_inbound./sum(Lint_Indicator_inbound);

%% encode per tetrode, conditioning on I=1 and I=0
procInd1_t1_Ia_out=procInd1_t1(ismember(procInd1_t1,ind_Indicator_outbound));
procInd1_t1_I_out=find(ismember(procInd1_t1,ind_Indicator_outbound));
Xnum_t1_I_out=normpdf(xs'*ones(1,length(procInd1_t1_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t1_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t1_I_out=sum(Xnum_t1_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t1_I_out=Lint_t1_I_out./sum(Lint_t1_I_out);
procInd1_t1_Ia_in=procInd1_t1(ismember(procInd1_t1,ind_Indicator_inbound));
procInd1_t1_I_in=find(ismember(procInd1_t1,ind_Indicator_inbound));
Xnum_t1_I_in=normpdf(xs'*ones(1,length(procInd1_t1_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t1_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t1_I_in=sum(Xnum_t1_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t1_I_in=Lint_t1_I_in./sum(Lint_t1_I_in);

procInd1_t2_Ia_out=procInd1_t2(ismember(procInd1_t2,ind_Indicator_outbound));
procInd1_t2_I_out=find(ismember(procInd1_t2,ind_Indicator_outbound));
Xnum_t2_I_out=normpdf(xs'*ones(1,length(procInd1_t2_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t2_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t2_I_out=sum(Xnum_t2_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t2_I_out=Lint_t2_I_out./sum(Lint_t2_I_out);
procInd1_t2_Ia_in=procInd1_t2(ismember(procInd1_t2,ind_Indicator_inbound));
procInd1_t2_I_in=find(ismember(procInd1_t2,ind_Indicator_inbound));
Xnum_t2_I_in=normpdf(xs'*ones(1,length(procInd1_t2_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t2_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t2_I_in=sum(Xnum_t2_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t2_I_in=Lint_t2_I_in./sum(Lint_t2_I_in);

procInd1_t4_Ia_out=procInd1_t4(ismember(procInd1_t4,ind_Indicator_outbound));
procInd1_t4_I_out=find(ismember(procInd1_t4,ind_Indicator_outbound));
Xnum_t4_I_out=normpdf(xs'*ones(1,length(procInd1_t4_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t4_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t4_I_out=sum(Xnum_t4_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t4_I_out=Lint_t4_I_out./sum(Lint_t4_I_out);
procInd1_t4_Ia_in=procInd1_t4(ismember(procInd1_t4,ind_Indicator_inbound));
procInd1_t4_I_in=find(ismember(procInd1_t4,ind_Indicator_inbound));
Xnum_t4_I_in=normpdf(xs'*ones(1,length(procInd1_t4_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t4_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t4_I_in=sum(Xnum_t4_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t4_I_in=Lint_t4_I_in./sum(Lint_t4_I_in);

procInd1_t5_Ia_out=procInd1_t5(ismember(procInd1_t5,ind_Indicator_outbound));
procInd1_t5_I_out=find(ismember(procInd1_t5,ind_Indicator_outbound));
Xnum_t5_I_out=normpdf(xs'*ones(1,length(procInd1_t5_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t5_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t5_I_out=sum(Xnum_t5_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t5_I_out=Lint_t5_I_out./sum(Lint_t5_I_out);
procInd1_t5_Ia_in=procInd1_t5(ismember(procInd1_t5,ind_Indicator_inbound));
procInd1_t5_I_in=find(ismember(procInd1_t5,ind_Indicator_inbound));
Xnum_t5_I_in=normpdf(xs'*ones(1,length(procInd1_t5_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t5_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t5_I_in=sum(Xnum_t5_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t5_I_in=Lint_t5_I_in./sum(Lint_t5_I_in);

procInd1_t7_Ia_out=procInd1_t7(ismember(procInd1_t7,ind_Indicator_outbound));
procInd1_t7_I_out=find(ismember(procInd1_t7,ind_Indicator_outbound));
Xnum_t7_I_out=normpdf(xs'*ones(1,length(procInd1_t7_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t7_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t7_I_out=sum(Xnum_t7_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t7_I_out=Lint_t7_I_out./sum(Lint_t7_I_out);
procInd1_t7_Ia_in=procInd1_t7(ismember(procInd1_t7,ind_Indicator_inbound));
procInd1_t7_I_in=find(ismember(procInd1_t7,ind_Indicator_inbound));
Xnum_t7_I_in=normpdf(xs'*ones(1,length(procInd1_t7_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t7_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t7_I_in=sum(Xnum_t7_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t7_I_in=Lint_t7_I_in./sum(Lint_t7_I_in);

procInd1_t10_Ia_out=procInd1_t10(ismember(procInd1_t10,ind_Indicator_outbound));
procInd1_t10_I_out=find(ismember(procInd1_t10,ind_Indicator_outbound));
Xnum_t10_I_out=normpdf(xs'*ones(1,length(procInd1_t10_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t10_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t10_I_out=sum(Xnum_t10_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t10_I_out=Lint_t10_I_out./sum(Lint_t10_I_out);
procInd1_t10_Ia_in=procInd1_t10(ismember(procInd1_t10,ind_Indicator_inbound));
procInd1_t10_I_in=find(ismember(procInd1_t10,ind_Indicator_inbound));
Xnum_t10_I_in=normpdf(xs'*ones(1,length(procInd1_t10_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t10_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t10_I_in=sum(Xnum_t10_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t10_I_in=Lint_t10_I_in./sum(Lint_t10_I_in);

procInd1_t11_Ia_out=procInd1_t11(ismember(procInd1_t11,ind_Indicator_outbound));
procInd1_t11_I_out=find(ismember(procInd1_t11,ind_Indicator_outbound));
Xnum_t11_I_out=normpdf(xs'*ones(1,length(procInd1_t11_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t11_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t11_I_out=sum(Xnum_t11_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t11_I_out=Lint_t11_I_out./sum(Lint_t11_I_out);
procInd1_t11_Ia_in=procInd1_t11(ismember(procInd1_t11,ind_Indicator_inbound));
procInd1_t11_I_in=find(ismember(procInd1_t11,ind_Indicator_inbound));
Xnum_t11_I_in=normpdf(xs'*ones(1,length(procInd1_t11_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t11_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t11_I_in=sum(Xnum_t11_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t11_I_in=Lint_t11_I_in./sum(Lint_t11_I_in);

procInd1_t12_Ia_out=procInd1_t12(ismember(procInd1_t12,ind_Indicator_outbound));
procInd1_t12_I_out=find(ismember(procInd1_t12,ind_Indicator_outbound));
Xnum_t12_I_out=normpdf(xs'*ones(1,length(procInd1_t12_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t12_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t12_I_out=sum(Xnum_t12_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t12_I_out=Lint_t12_I_out./sum(Lint_t12_I_out);
procInd1_t12_Ia_in=procInd1_t12(ismember(procInd1_t12,ind_Indicator_inbound));
procInd1_t12_I_in=find(ismember(procInd1_t12,ind_Indicator_inbound));
Xnum_t12_I_in=normpdf(xs'*ones(1,length(procInd1_t12_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t12_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t12_I_in=sum(Xnum_t12_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t12_I_in=Lint_t12_I_in./sum(Lint_t12_I_in);

procInd1_t13_Ia_out=procInd1_t13(ismember(procInd1_t13,ind_Indicator_outbound));
procInd1_t13_I_out=find(ismember(procInd1_t13,ind_Indicator_outbound));
Xnum_t13_I_out=normpdf(xs'*ones(1,length(procInd1_t13_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t13_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t13_I_out=sum(Xnum_t13_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t13_I_out=Lint_t13_I_out./sum(Lint_t13_I_out);
procInd1_t13_Ia_in=procInd1_t13(ismember(procInd1_t13,ind_Indicator_inbound));
procInd1_t13_I_in=find(ismember(procInd1_t13,ind_Indicator_inbound));
Xnum_t13_I_in=normpdf(xs'*ones(1,length(procInd1_t13_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t13_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t13_I_in=sum(Xnum_t13_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t13_I_in=Lint_t13_I_in./sum(Lint_t13_I_in);

procInd1_t14_Ia_out=procInd1_t14(ismember(procInd1_t14,ind_Indicator_outbound));
procInd1_t14_I_out=find(ismember(procInd1_t14,ind_Indicator_outbound));
Xnum_t14_I_out=normpdf(xs'*ones(1,length(procInd1_t14_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t14_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t14_I_out=sum(Xnum_t14_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t14_I_out=Lint_t14_I_out./sum(Lint_t14_I_out);
procInd1_t14_Ia_in=procInd1_t14(ismember(procInd1_t14,ind_Indicator_inbound));
procInd1_t14_I_in=find(ismember(procInd1_t14,ind_Indicator_inbound));
Xnum_t14_I_in=normpdf(xs'*ones(1,length(procInd1_t14_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t14_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t14_I_in=sum(Xnum_t14_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t14_I_in=Lint_t14_I_in./sum(Lint_t14_I_in);

procInd1_t17_Ia_out=procInd1_t17(ismember(procInd1_t17,ind_Indicator_outbound));
procInd1_t17_I_out=find(ismember(procInd1_t17,ind_Indicator_outbound));
Xnum_t17_I_out=normpdf(xs'*ones(1,length(procInd1_t17_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t17_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t17_I_out=sum(Xnum_t17_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t17_I_out=Lint_t17_I_out./sum(Lint_t17_I_out);
procInd1_t17_Ia_in=procInd1_t17(ismember(procInd1_t17,ind_Indicator_inbound));
procInd1_t17_I_in=find(ismember(procInd1_t17,ind_Indicator_inbound));
Xnum_t17_I_in=normpdf(xs'*ones(1,length(procInd1_t17_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t17_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t17_I_in=sum(Xnum_t17_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t17_I_in=Lint_t17_I_in./sum(Lint_t17_I_in);

procInd1_t18_Ia_out=procInd1_t18(ismember(procInd1_t18,ind_Indicator_outbound));
procInd1_t18_I_out=find(ismember(procInd1_t18,ind_Indicator_outbound));
Xnum_t18_I_out=normpdf(xs'*ones(1,length(procInd1_t18_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t18_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t18_I_out=sum(Xnum_t18_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t18_I_out=Lint_t18_I_out./sum(Lint_t18_I_out);
procInd1_t18_Ia_in=procInd1_t18(ismember(procInd1_t18,ind_Indicator_inbound));
procInd1_t18_I_in=find(ismember(procInd1_t18,ind_Indicator_inbound));
Xnum_t18_I_in=normpdf(xs'*ones(1,length(procInd1_t18_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t18_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t18_I_in=sum(Xnum_t18_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t18_I_in=Lint_t18_I_in./sum(Lint_t18_I_in);

procInd1_t19_Ia_out=procInd1_t19(ismember(procInd1_t19,ind_Indicator_outbound));
procInd1_t19_I_out=find(ismember(procInd1_t19,ind_Indicator_outbound));
Xnum_t19_I_out=normpdf(xs'*ones(1,length(procInd1_t19_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t19_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t19_I_out=sum(Xnum_t19_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t19_I_out=Lint_t19_I_out./sum(Lint_t19_I_out);
procInd1_t19_Ia_in=procInd1_t19(ismember(procInd1_t19,ind_Indicator_outbound));
procInd1_t19_I_in=find(ismember(procInd1_t19,ind_Indicator_outbound));
Xnum_t19_I_in=normpdf(xs'*ones(1,length(procInd1_t19_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t19_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t19_I_in=sum(Xnum_t19_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t19_I_in=Lint_t19_I_in./sum(Lint_t19_I_in);

procInd1_t20_Ia_out=procInd1_t20(ismember(procInd1_t20,ind_Indicator_outbound));
procInd1_t20_I_out=find(ismember(procInd1_t20,ind_Indicator_outbound));
Xnum_t20_I_out=normpdf(xs'*ones(1,length(procInd1_t20_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t20_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t20_I_out=sum(Xnum_t20_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t20_I_out=Lint_t20_I_out./sum(Lint_t20_I_out);
procInd1_t20_Ia_in=procInd1_t20(ismember(procInd1_t20,ind_Indicator_inbound));
procInd1_t20_I_in=find(ismember(procInd1_t20,ind_Indicator_inbound));
Xnum_t20_I_in=normpdf(xs'*ones(1,length(procInd1_t20_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t20_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t20_I_in=sum(Xnum_t20_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t20_I_in=Lint_t20_I_in./sum(Lint_t20_I_in);

procInd1_t22_Ia_out=procInd1_t22(ismember(procInd1_t22,ind_Indicator_outbound));
procInd1_t22_I_out=find(ismember(procInd1_t22,ind_Indicator_outbound));
Xnum_t22_I_out=normpdf(xs'*ones(1,length(procInd1_t22_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t22_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t22_I_out=sum(Xnum_t22_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t22_I_out=Lint_t22_I_out./sum(Lint_t22_I_out);
procInd1_t22_Ia_in=procInd1_t22(ismember(procInd1_t22,ind_Indicator_outbound));
procInd1_t22_I_in=find(ismember(procInd1_t22,ind_Indicator_outbound));
Xnum_t22_I_in=normpdf(xs'*ones(1,length(procInd1_t22_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t22_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t22_I_in=sum(Xnum_t22_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t22_I_in=Lint_t22_I_in./sum(Lint_t22_I_in);

procInd1_t23_Ia_out=procInd1_t23(ismember(procInd1_t23,ind_Indicator_outbound));
procInd1_t23_I_out=find(ismember(procInd1_t23,ind_Indicator_outbound));
Xnum_t23_I_out=normpdf(xs'*ones(1,length(procInd1_t23_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t23_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t23_I_out=sum(Xnum_t23_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t23_I_out=Lint_t23_I_out./sum(Lint_t23_I_out);
procInd1_t23_Ia_in=procInd1_t23(ismember(procInd1_t23,ind_Indicator_inbound));
procInd1_t23_I_in=find(ismember(procInd1_t23,ind_Indicator_inbound));
Xnum_t23_I_in=normpdf(xs'*ones(1,length(procInd1_t23_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t23_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t23_I_in=sum(Xnum_t23_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t23_I_in=Lint_t23_I_in./sum(Lint_t23_I_in);

procInd1_t27_Ia_out=procInd1_t27(ismember(procInd1_t27,ind_Indicator_outbound));
procInd1_t27_I_out=find(ismember(procInd1_t27,ind_Indicator_outbound));
Xnum_t27_I_out=normpdf(xs'*ones(1,length(procInd1_t27_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t27_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t27_I_out=sum(Xnum_t27_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t27_I_out=Lint_t27_I_out./sum(Lint_t27_I_out);
procInd1_t27_Ia_in=procInd1_t27(ismember(procInd1_t27,ind_Indicator_inbound));
procInd1_t27_I_in=find(ismember(procInd1_t27,ind_Indicator_inbound));
Xnum_t27_I_in=normpdf(xs'*ones(1,length(procInd1_t27_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t27_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t27_I_in=sum(Xnum_t27_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t27_I_in=Lint_t27_I_in./sum(Lint_t27_I_in);

procInd1_t29_Ia_out=procInd1_t29(ismember(procInd1_t29,ind_Indicator_outbound));
procInd1_t29_I_out=find(ismember(procInd1_t29,ind_Indicator_outbound));
Xnum_t29_I_out=normpdf(xs'*ones(1,length(procInd1_t29_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t29_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t29_I_out=sum(Xnum_t29_I_out,2)./occ_Indicator_outbound(:,1)./dt; %integral
Lint_t29_I_out=Lint_t29_I_out./sum(Lint_t29_I_out);
procInd1_t29_Ia_in=procInd1_t29(ismember(procInd1_t29,ind_Indicator_inbound));
procInd1_t29_I_in=find(ismember(procInd1_t29,ind_Indicator_inbound));
Xnum_t29_I_in=normpdf(xs'*ones(1,length(procInd1_t29_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t29_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t29_I_in=sum(Xnum_t29_I_in,2)./occ_Indicator_inbound(:,1)./dt; %integral
Lint_t29_I_in=Lint_t29_I_in./sum(Lint_t29_I_in);
end