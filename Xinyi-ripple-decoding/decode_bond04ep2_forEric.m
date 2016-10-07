
%%
%%%%bond04, epoch 2
load('bond_data/bontask04.mat');
load('bond_data/bonpos04.mat');
load('bond_data/bonspikes04.mat'); %sorted spikes
load('bond_data/bonlinpos04.mat');
load('bond_data/bonripplescons04.mat');
load('bond_data/bontrajencode04.mat');
load('bond_data/bonwellvisits04.mat');

ex=4; ep=2;

ind_t=[1 2 4 5 7 10 11 12 13 14 17 18 19 20 22 23 27 29]; %tetrode index
a=[];b=[]; 
%a: tetrode index; b: corresponding cell index
%this is from sorted spikes, we only use this to select replay events with
%non-zero sorted spikes in it to decode
for ind=1:length(ind_t)
    numC=length(spikes{ex}{ep}{ind_t(ind)});
    b0=zeros(1,numC);a0=zeros(1,numC);
    if isempty(spikes{ex}{ep}{ind_t(ind)})==0
    for i=1:numC
        if isempty(spikes{ex}{ep}{ind_t(ind)}{i})==0 && isempty(spikes{ex}{ep}{ind_t(ind)}{i}.data)==0
            b0(i)=i;
            a0(i)=ind_t(ind);
        end
    end
    end
    b=[b b0(find(b0))];a=[a a0(find(a0))];
end



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%       ENCODE             %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% use Loren's linearization
time=linpos{ex}{ep}.statematrix.time;
lindist=linpos{ex}{ep}.statematrix.lindist;
vecLF(:,1)=time;vecLF(:,2)=lindist;
%figure;plot(time,lindist,'.');

A=pos{ex}{ep}.data(:,1); %time stamps for animal's trajectory
ti=round(A(1)*1000):1:round(A(end)*1000); %binning time stamps at 1 ms
stateV=linspace(min(lindist),max(lindist),61);
xdel=stateV(2)-stateV(1);


%% calculate emipirical movement transition matrix, then Gaussian smoothed
[sn,state_bin]=histc(lindist,stateV);
state_disM=[state_bin(1:end-1) state_bin(2:end)];
n=size(stateV,2);
%by column=departuring
for s=1:n
    sp0=state_disM(find(state_disM(:,1)==s),2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
    if isempty(sp0)==0
        stateM(:,s)=histc(sp0,linspace(1,n,n))./size(sp0,1);
    elseif isempty(sp0)==1
        stateM(:,s)=zeros(1,n);
    end
end
K=inline('exp(-(x.^2+y.^2)/2/sig^2)'); %gaussian
[dx,dy]=meshgrid([-1:1]);
sig=0.5;
weight=K(sig,dx,dy)/sum(sum(K(sig,dx,dy))); %normalizing weights
stateM_gaus=conv2(stateM,weight,'same'); %gaussian smoothed
stateM_gausnorm=stateM_gaus*diag(1./sum(stateM_gaus,1)); %normalized to confine probability to 1

%%
ind_I_out=find(trajencode{ex}{ep}.trajstate==1 | trajencode{ex}{ep}.trajstate==3);
ind_I_in=find(trajencode{ex}{ep}.trajstate==2 | trajencode{ex}{ep}.trajstate==4);
%figure;plot(vecLF(ind_I_out,1),vecLF(ind_I_out,2),'r.',vecLF(ind_I_in,1),vecLF(ind_I_in,2),'b.');

%% empirical movement transition matrix conditioned on I=1(outbound) and I=0 (inbound)
n=length(stateV);
stateM_I_out=zeros(n,n);
vecLF_seg=vecLF(ind_I_out,:);
[sn,state_bin]=histc(vecLF_seg(:,2),stateV);
state_disM=[state_bin(1:end-1) state_bin(2:end)];
stateM_seg=zeros(n,n);
    for s=1:n
        sp0=state_disM(find(state_disM(:,1)==s),2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
        if isempty(sp0)==0
            stateM_seg(:,s)=histc(sp0,linspace(1,n,n))./size(sp0,1);
        elseif isempty(sp0)==1
            stateM_seg(:,s)=zeros(1,n);
        end
    end
stateM_I_out=stateM_I_out+stateM_seg;
%%%if too many zeros:
for i=1:n
    if sum(stateM_I_out(:,i))==0
        stateM_I_out(:,i)=1/n;
    elseif sum(stateM_I_out(:,i))~=0
        stateM_I_out(:,i)=stateM_I_out(:,i)./sum(stateM_I_out(:,i));
    end
end
%stateM_I1=stateM_I1*diag(1./sum(stateM_I1,1));
K=inline('exp(-(x.^2+y.^2)/2/sig^2)'); %gaussian
[dx,dy]=meshgrid([-1:1]);
sig=0.5;
weight=K(sig,dx,dy)/sum(sum(K(sig,dx,dy))); %normalizing weights
stateM_gaus=conv2(stateM_I_out,weight,'same'); %gaussian smoothed
stateM_I1_gausnorm=stateM_gaus*diag(1./sum(stateM_gaus,1)); %normalized to confine probability to 1


stateM_I_in=zeros(n,n);
vecLF_seg=vecLF(ind_I_in,:);
[sn,state_bin]=histc(vecLF_seg(:,2),stateV);
state_disM=[state_bin(1:end-1) state_bin(2:end)];
stateM_seg=zeros(n,n);
    for s=1:n
        sp0=state_disM(find(state_disM(:,1)==s),2); %by departure x_k-1 (by column); sp0 is the departuring x_(k-1);
        if isempty(sp0)==0
            stateM_seg(:,s)=histc(sp0,linspace(1,n,n))./size(sp0,1);
        elseif isempty(sp0)==1
            stateM_seg(:,s)=zeros(1,n);
        end
    end
stateM_I_in=stateM_I_in+stateM_seg;
%%if too many zeros
for i=1:n
    if sum(stateM_I_in(:,i))==0
        stateM_I_in(:,i)=1/n;
    elseif sum(stateM_I_in(:,i))~=0
        stateM_I_in(:,i)=stateM_I_in(:,i)./sum(stateM_I_in(:,i));
    end
end
%stateM_I0=stateM_I0*diag(1./sum(stateM_I0,1));
K=inline('exp(-(x.^2+y.^2)/2/sig^2)'); %gaussian
[dx,dy]=meshgrid([-1:1]);
sig=0.5;
weight=K(sig,dx,dy)/sum(sum(K(sig,dx,dy))); %normalizing weights
stateM_gaus=conv2(stateM_I_in,weight,'same'); %gaussian smoothed
stateM_I0_gausnorm=stateM_gaus*diag(1./sum(stateM_gaus,1)); %normalized to confine probability to 1


%% calculate ripple starting and end times
startT=ripplescons{ex}{ep}{1}.starttime;
endT=ripplescons{ex}{ep}{1}.endtime;
traj_Ind=find(ripplescons{ex}{ep}{1}.maxthresh>4);
startT=startT(traj_Ind);
endT=endT(traj_Ind);
ripple_seg=[round(startT*1000)-ti(1)-1 round(endT*1000)-ti(1)-1]; %index for ripple segments

clear sptrain2_list;
for kk=1:size(a,2)
    j=a(kk);i=b(kk);
    B=spikes{ex}{ep}{j}{i}.data(:,1); %spiking times for tetrode j, cell i
    xi=round(B*1000); %binning spiking times at 1 ms
    [sptrain2,~]=ismember(ti,xi); %sptrain2: spike train binned at 1 ms instead of 33.4ms (sptrain0)
    sptrain2_list{kk}=sptrain2;
end

clear spike_r_all;
for k=1:size(ripple_seg,1)
    spike_r=[];
    for kk=1:size(a,2)
        sptrain2=sptrain2_list{kk};
        spike_r=[spike_r;sptrain2(ripple_seg(k,1):ripple_seg(k,2))];
    end
    spike_r_all{k}=spike_r;
end

clear sumR;
for k=1:size(ripple_seg,1)
    spike_r=spike_r_all{k};
    sumR(k)=sum(spike_r(:));
end
rippleI=find(sumR>0);length(rippleI)

%% prepare kernel density model
tlin=linpos{ex}{ep}.statematrix.time;

poslin=vecLF(:,2);
xs=min(poslin):xdel:max(poslin);
dt=tlin(2)-tlin(1);
xtrain=poslin';

sxker=xdel; mdel=20; smker=mdel; T=size(tlin,1);

%% encode the kernel density model per tetrode
load('bond_data\bond04-01_params.mat');
%ind_t1=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t1=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t1=filedata.params(ind_t1,1);
mark0_t1=[filedata.params(ind_t1,2) filedata.params(ind_t1,3) filedata.params(ind_t1,4) filedata.params(ind_t1,5)];
time2_t1=time_t1/10000;
spikeT0_t1=time2_t1;
[procInd0_t1,procInd1_t1]=histc(spikeT0_t1,tlin);
procInd_t1=find(procInd0_t1);
spikeT_t1=tlin(procInd_t1);
spike_t1=procInd0_t1';
[~,rawInd0_t1]=histc(spikeT0_t1,time2_t1);
markAll_t1(:,1)=procInd1_t1;markAll_t1(:,2:5)=mark0_t1(rawInd0_t1(rawInd0_t1~=0),:);
ms=min(min(markAll_t1(:,2:5))):mdel:max(max(markAll_t1(:,2:5))); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t1=normpdf(xs'*ones(1,length(spikeT0_t1)),ones(length(xs),1)*xtrain(procInd1_t1),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t1=sum(Xnum_t1,2)./occ(:,1)./dt; %integral
Lint_t1=Lint_t1./sum(Lint_t1);

load('bond_data\bond04\bond04-02_params.mat');
%ind_t2=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t2=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t2=filedata.params(ind_t2,1);
mark0_t2=[filedata.params(ind_t2,2) filedata.params(ind_t2,3) filedata.params(ind_t2,4) filedata.params(ind_t2,5)];
time2_t2=time_t2/10000;
spikeT0_t2=time2_t2;
[procInd0_t2,procInd1_t2]=histc(spikeT0_t2,tlin);
procInd_t2=find(procInd0_t2);
spikeT_t2=tlin(procInd_t2);
spike_t2=procInd0_t2';
[~,rawInd0_t2]=histc(spikeT0_t2,time2_t2);
markAll_t2(:,1)=procInd1_t2;markAll_t2(:,2:5)=mark0_t2(rawInd0_t2(rawInd0_t2~=0),:);
ms=min(min(markAll_t2(:,2:5))):mdel:max(max(markAll_t2(:,2:5))); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t2=normpdf(xs'*ones(1,length(spikeT0_t2)),ones(length(xs),1)*xtrain(procInd1_t2),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t2=sum(Xnum_t2,2)./occ(:,1)./dt; %integral
Lint_t2=Lint_t2./sum(Lint_t2);

load('bond_data\bond04-04_params.mat');
%ind_t4=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t4=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t4=filedata.params(ind_t4,1);
mark0_t4=[filedata.params(ind_t4,2) filedata.params(ind_t4,3) filedata.params(ind_t4,4) filedata.params(ind_t4,5)];
time2_t4=time_t4/10000;
spikeT0_t4=time2_t4;
[procInd0_t4,procInd1_t4]=histc(spikeT0_t4,tlin);
procInd_t4=find(procInd0_t4);
spikeT_t4=tlin(procInd_t4);
spike_t4=procInd0_t4';
[~,rawInd0_t4]=histc(spikeT0_t4,time2_t4);
markAll_t4(:,1)=procInd1_t4;markAll_t4(:,2:5)=mark0_t4(rawInd0_t4(rawInd0_t4~=0),:);
ms=min(min(markAll_t4(:,2:5))):mdel:max(max(markAll_t4(:,2:5))); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t4=normpdf(xs'*ones(1,length(spikeT0_t4)),ones(length(xs),1)*xtrain(procInd1_t4),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t4=sum(Xnum_t4,2)./occ(:,1)./dt; %integral
Lint_t4=Lint_t4./sum(Lint_t4);

load('bond_data\bond04-05_params.mat');
%ind_t5=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t5=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t5=filedata.params(ind_t5,1);
mark0_t5=[filedata.params(ind_t5,2) filedata.params(ind_t5,3) filedata.params(ind_t5,4) filedata.params(ind_t5,5)];
time2_t5=time_t5/10000;
spikeT0_t5=time2_t5;
[procInd0_t5,procInd1_t5]=histc(spikeT0_t5,tlin);
procInd_t5=find(procInd0_t5);
spikeT_t5=tlin(procInd_t5);
spike_t5=procInd0_t5';
[~,rawInd0_t5]=histc(spikeT0_t5,time2_t5);
markAll_t5(:,1)=procInd1_t5;markAll_t5(:,2:5)=mark0_t5(rawInd0_t5(rawInd0_t5~=0),:);
ms=min(min(markAll_t5(:,2:5))):mdel:max(max(markAll_t5(:,2:5))); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t5=normpdf(xs'*ones(1,length(spikeT0_t5)),ones(length(xs),1)*xtrain(procInd1_t5),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t5=sum(Xnum_t5,2)./occ(:,1)./dt; %integral
Lint_t5=Lint_t5./sum(Lint_t5);

load('bond_data\bond04-07_params.mat');
%ind_t7=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t7=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t7=filedata.params(ind_t7,1);
mark0_t7=[filedata.params(ind_t7,2) filedata.params(ind_t7,3) filedata.params(ind_t7,4) filedata.params(ind_t7,5)];
time2_t7=time_t7/10000;
spikeT0_t7=time2_t7;
[procInd0_t7,procInd1_t7]=histc(spikeT0_t7,tlin);
procInd_t7=find(procInd0_t7);
spikeT_t7=tlin(procInd_t7);
spike_t7=procInd0_t7';
[~,rawInd0_t7]=histc(spikeT0_t7,time2_t7);
markAll_t7(:,1)=procInd1_t7;markAll_t7(:,2:5)=mark0_t7(rawInd0_t7(rawInd0_t7~=0),:);
ms=min(min(markAll_t7(:,2:5))):mdel:max(max(markAll_t7(:,2:5))); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t7=normpdf(xs'*ones(1,length(spikeT0_t7)),ones(length(xs),1)*xtrain(procInd1_t7),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t7=sum(Xnum_t7,2)./occ(:,1)./dt; %integral
Lint_t7=Lint_t7./sum(Lint_t7);

load('bond_data\bond04-10_params.mat');
%ind_t10=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t10=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t10=filedata.params(ind_t10,1);
mark0_t10=[filedata.params(ind_t10,2) filedata.params(ind_t10,3) filedata.params(ind_t10,4) filedata.params(ind_t10,5)];
time2_t10=time_t10/10000;
spikeT0_t10=time2_t10;
[procInd0_t10,procInd1_t10]=histc(spikeT0_t10,tlin);
procInd_t10=find(procInd0_t10);
spikeT_t10=tlin(procInd_t10);
spike_t10=procInd0_t10';
[~,rawInd0_t10]=histc(spikeT0_t10,time2_t10);
markAll_t10(:,1)=procInd1_t10;markAll_t10(:,2:5)=mark0_t10(rawInd0_t10(rawInd0_t10~=0),:);
ms=min(min(markAll_t10(:,2:5))):mdel:max(max(markAll_t10(:,2:5))); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t10=normpdf(xs'*ones(1,length(spikeT0_t10)),ones(length(xs),1)*xtrain(procInd1_t10),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t10=sum(Xnum_t10,2)./occ(:,1)./dt; %integral
Lint_t10=Lint_t10./sum(Lint_t10);

load('bond_data\bond04-11_params.mat');
%ind_t11=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t11=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t11=filedata.params(ind_t11,1);
mark0_t11=[filedata.params(ind_t11,2) filedata.params(ind_t11,3) filedata.params(ind_t11,4) filedata.params(ind_t11,5)];
time2_t11=time_t11/10000;
spikeT0_t11=time2_t11;
[procInd0_t11,procInd1_t11]=histc(spikeT0_t11,tlin);
procInd_t11=find(procInd0_t11);
spikeT_t11=tlin(procInd_t11);
spike_t11=procInd0_t11';
[~,rawInd0_t11]=histc(spikeT0_t11,time2_t11);
markAll_t11(:,1)=procInd1_t11;markAll_t11(:,2:5)=mark0_t11(rawInd0_t11(rawInd0_t11~=0),:);
ms=min(min(markAll_t11(:,2:5))):mdel:max(max(markAll_t11(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t11=normpdf(xs'*ones(1,length(spikeT0_t11)),ones(length(xs),1)*xtrain(procInd1_t11),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t11=sum(Xnum_t11,2)./occ(:,1)./dt; %integral
Lint_t11=Lint_t11./sum(Lint_t11);

load('bond_data\bond04-12_params.mat');
%ind_t12=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t12=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t12=filedata.params(ind_t12,1);
mark0_t12=[filedata.params(ind_t12,2) filedata.params(ind_t12,3) filedata.params(ind_t12,4) filedata.params(ind_t12,5)];
time2_t12=time_t12/10000;
spikeT0_t12=time2_t12;
[procInd0_t12,procInd1_t12]=histc(spikeT0_t12,tlin);
procInd_t12=find(procInd0_t12);
spikeT_t12=tlin(procInd_t12);
spike_t12=procInd0_t12';
[~,rawInd0_t12]=histc(spikeT0_t12,time2_t12);
markAll_t12(:,1)=procInd1_t12;markAll_t12(:,2:5)=mark0_t12(rawInd0_t12(rawInd0_t12~=0),:);
ms=min(min(markAll_t12(:,2:5))):mdel:max(max(markAll_t12(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t12=normpdf(xs'*ones(1,length(spikeT0_t12)),ones(length(xs),1)*xtrain(procInd1_t12),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t12=sum(Xnum_t12,2)./occ(:,1)./dt; %integral
Lint_t12=Lint_t12./sum(Lint_t12);

load('bond_data\bond04-13_params.mat');
%ind_t13=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t13=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t13=filedata.params(ind_t13,1);
mark0_t13=[filedata.params(ind_t13,2) filedata.params(ind_t13,3) filedata.params(ind_t13,4) filedata.params(ind_t13,5)];
time2_t13=time_t13/10000;
spikeT0_t13=time2_t13;
[procInd0_t13,procInd1_t13]=histc(spikeT0_t13,tlin);
procInd_t13=find(procInd0_t13);
spikeT_t13=tlin(procInd_t13);
spike_t13=procInd0_t13';
[~,rawInd0_t13]=histc(spikeT0_t13,time2_t13);
markAll_t13(:,1)=procInd1_t13;markAll_t13(:,2:5)=mark0_t13(rawInd0_t13(rawInd0_t13~=0),:);
ms=min(min(markAll_t13(:,2:5))):mdel:max(max(markAll_t13(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t13=normpdf(xs'*ones(1,length(spikeT0_t13)),ones(length(xs),1)*xtrain(procInd1_t13),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t13=sum(Xnum_t13,2)./occ(:,1)./dt; %integral
Lint_t13=Lint_t13./sum(Lint_t13);

load('bond_data\bond04-14_params.mat');
%ind_t14=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t14=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t14=filedata.params(ind_t14,1);
mark0_t14=[filedata.params(ind_t14,2) filedata.params(ind_t14,3) filedata.params(ind_t14,4) filedata.params(ind_t14,5)];
time2_t14=time_t14/10000;
spikeT0_t14=time2_t14;
[procInd0_t14,procInd1_t14]=histc(spikeT0_t14,tlin);
procInd_t14=find(procInd0_t14);
spikeT_t14=tlin(procInd_t14);
spike_t14=procInd0_t14';
[~,rawInd0_t14]=histc(spikeT0_t14,time2_t14);
markAll_t14(:,1)=procInd1_t14;markAll_t14(:,2:5)=mark0_t14(rawInd0_t14(rawInd0_t14~=0),:);
ms=min(min(markAll_t14(:,2:5))):mdel:max(max(markAll_t14(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t14=normpdf(xs'*ones(1,length(spikeT0_t14)),ones(length(xs),1)*xtrain(procInd1_t14),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t14=sum(Xnum_t14,2)./occ(:,1)./dt; %integral
Lint_t14=Lint_t14./sum(Lint_t14);

load('bond_data\bond04-17_params.mat');
%ind_t17=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t17=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t17=filedata.params(ind_t17,1);
mark0_t17=[filedata.params(ind_t17,2) filedata.params(ind_t17,3) filedata.params(ind_t17,4) filedata.params(ind_t17,5)];
time2_t17=time_t17/10000;
spikeT0_t17=time2_t17;
[procInd0_t17,procInd1_t17]=histc(spikeT0_t17,tlin);
procInd_t17=find(procInd0_t17);
spikeT_t17=tlin(procInd_t17);
spike_t17=procInd0_t17';
[~,rawInd0_t17]=histc(spikeT0_t17,time2_t17);
markAll_t17(:,1)=procInd1_t17;markAll_t17(:,2:5)=mark0_t17(rawInd0_t17(rawInd0_t17~=0),:);
ms=min(min(markAll_t17(:,2:5))):mdel:max(max(markAll_t17(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t17=normpdf(xs'*ones(1,length(spikeT0_t17)),ones(length(xs),1)*xtrain(procInd1_t17),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t17=sum(Xnum_t17,2)./occ(:,1)./dt; %integral
Lint_t17=Lint_t17./sum(Lint_t17);

load('bond_data\bond04-18_params.mat');
%ind_t18=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t18=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t18=filedata.params(ind_t18,1);
mark0_t18=[filedata.params(ind_t18,2) filedata.params(ind_t18,3) filedata.params(ind_t18,4) filedata.params(ind_t18,5)];
time2_t18=time_t18/10000;
spikeT0_t18=time2_t18;
[procInd0_t18,procInd1_t18]=histc(spikeT0_t18,tlin);
procInd_t18=find(procInd0_t18);
spikeT_t18=tlin(procInd_t18);
spike_t18=procInd0_t18';
[~,rawInd0_t18]=histc(spikeT0_t18,time2_t18);
markAll_t18(:,1)=procInd1_t18;markAll_t18(:,2:5)=mark0_t18(rawInd0_t18(rawInd0_t18~=0),:);
ms=min(min(markAll_t18(:,2:5))):mdel:max(max(markAll_t18(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t18=normpdf(xs'*ones(1,length(spikeT0_t18)),ones(length(xs),1)*xtrain(procInd1_t18),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t18=sum(Xnum_t18,2)./occ(:,1)./dt; %integral
Lint_t18=Lint_t18./sum(Lint_t18);

load('bond_data\bond04-19_params.mat');
%ind_t19=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t19=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t19=filedata.params(ind_t19,1);
mark0_t19=[filedata.params(ind_t19,2) filedata.params(ind_t19,3) filedata.params(ind_t19,4) filedata.params(ind_t19,5)];
time2_t19=time_t19/10000;
spikeT0_t19=time2_t19;
[procInd0_t19,procInd1_t19]=histc(spikeT0_t19,tlin);
procInd_t19=find(procInd0_t19);
spikeT_t19=tlin(procInd_t19);
spike_t19=procInd0_t19';
[~,rawInd0_t19]=histc(spikeT0_t19,time2_t19);
markAll_t19(:,1)=procInd1_t19;markAll_t19(:,2:5)=mark0_t19(rawInd0_t19(rawInd0_t19~=0),:);
ms=min(min(markAll_t19(:,2:5))):mdel:max(max(markAll_t19(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t19=normpdf(xs'*ones(1,length(spikeT0_t19)),ones(length(xs),1)*xtrain(procInd1_t19),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t19=sum(Xnum_t19,2)./occ(:,1)./dt; %integral
Lint_t19=Lint_t19./sum(Lint_t19);

load('bond_data\bond04-20_params.mat');
%ind_t20=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t20=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t20=filedata.params(ind_t20,1);
mark0_t20=[filedata.params(ind_t20,2) filedata.params(ind_t20,3) filedata.params(ind_t20,4) filedata.params(ind_t20,5)];
time2_t20=time_t20/10000;
spikeT0_t20=time2_t20;
[procInd0_t20,procInd1_t20]=histc(spikeT0_t20,tlin);
procInd_t20=find(procInd0_t20);
spikeT_t20=tlin(procInd_t20);
spike_t20=procInd0_t20';
[~,rawInd0_t20]=histc(spikeT0_t20,time2_t20);
markAll_t20(:,1)=procInd1_t20;markAll_t20(:,2:5)=mark0_t20(rawInd0_t20(rawInd0_t20~=0),:);
ms=min(min(markAll_t20(:,2:5))):mdel:max(max(markAll_t20(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t20=normpdf(xs'*ones(1,length(spikeT0_t20)),ones(length(xs),1)*xtrain(procInd1_t20),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t20=sum(Xnum_t20,2)./occ(:,1)./dt; %integral
Lint_t20=Lint_t20./sum(Lint_t20);

load('bond_data\bond04-22_params.mat');
%ind_t22=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t22=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t22=filedata.params(ind_t22,1);
mark0_t22=[filedata.params(ind_t22,2) filedata.params(ind_t22,3) filedata.params(ind_t22,4) filedata.params(ind_t22,5)];
time2_t22=time_t22/10000;
spikeT0_t22=time2_t22;
[procInd0_t22,procInd1_t22]=histc(spikeT0_t22,tlin);
procInd_t22=find(procInd0_t22);
spikeT_t22=tlin(procInd_t22);
spike_t22=procInd0_t22';
[~,rawInd0_t22]=histc(spikeT0_t22,time2_t22);
markAll_t22(:,1)=procInd1_t22;markAll_t22(:,2:5)=mark0_t22(rawInd0_t22(rawInd0_t22~=0),:);
ms=min(min(markAll_t22(:,2:5))):mdel:max(max(markAll_t22(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t22=normpdf(xs'*ones(1,length(spikeT0_t22)),ones(length(xs),1)*xtrain(procInd1_t22),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t22=sum(Xnum_t22,2)./occ(:,1)./dt; %integral
Lint_t22=Lint_t22./sum(Lint_t22);

load('bond_data\bond04-23_params.mat');
%ind_t23=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t23=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t23=filedata.params(ind_t23,1);
mark0_t23=[filedata.params(ind_t23,2) filedata.params(ind_t23,3) filedata.params(ind_t23,4) filedata.params(ind_t23,5)];
time2_t23=time_t23/10000;
spikeT0_t23=time2_t23;
[procInd0_t23,procInd1_t23]=histc(spikeT0_t23,tlin);
procInd_t23=find(procInd0_t23);
spikeT_t23=tlin(procInd_t23);
spike_t23=procInd0_t23';
[~,rawInd0_t23]=histc(spikeT0_t23,time2_t23);
markAll_t23(:,1)=procInd1_t23;markAll_t23(:,2:5)=mark0_t23(rawInd0_t23(rawInd0_t23~=0),:);
ms=min(min(markAll_t23(:,2:5))):mdel:max(max(markAll_t23(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t23=normpdf(xs'*ones(1,length(spikeT0_t23)),ones(length(xs),1)*xtrain(procInd1_t23),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t23=sum(Xnum_t23,2)./occ(:,1)./dt; %integral
Lint_t23=Lint_t23./sum(Lint_t23);

load('bond_data\bond04-27_params.mat');
%ind_t27=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t27=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t27=filedata.params(ind_t27,1);
mark0_t27=[filedata.params(ind_t27,2) filedata.params(ind_t27,3) filedata.params(ind_t27,4) filedata.params(ind_t27,5)];
time2_t27=time_t27/10000;
spikeT0_t27=time2_t27;
[procInd0_t27,procInd1_t27]=histc(spikeT0_t27,tlin);
procInd_t27=find(procInd0_t27);
spikeT_t27=tlin(procInd_t27);
spike_t27=procInd0_t27';
[~,rawInd0_t27]=histc(spikeT0_t27,time2_t27);
markAll_t27(:,1)=procInd1_t27;markAll_t27(:,2:5)=mark0_t27(rawInd0_t27(rawInd0_t27~=0),:);
ms=min(min(markAll_t27(:,2:5))):mdel:max(max(markAll_t27(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t27=normpdf(xs'*ones(1,length(spikeT0_t27)),ones(length(xs),1)*xtrain(procInd1_t27),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t27=sum(Xnum_t27,2)./occ(:,1)./dt; %integral
Lint_t27=Lint_t27./sum(Lint_t27);

load('bond_data\bond04-29_params.mat');
%ind_t29=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end)&filedata.params(:,8)>10&(filedata.params(:,2)>100|filedata.params(:,3)>100|filedata.params(:,4)>100|filedata.params(:,5)>100));
ind_t29=find(filedata.params(:,1)/10000>=tlin(1)&filedata.params(:,1)/10000<=tlin(end));
time_t29=filedata.params(ind_t29,1);
mark0_t29=[filedata.params(ind_t29,2) filedata.params(ind_t29,3) filedata.params(ind_t29,4) filedata.params(ind_t29,5)];
time2_t29=time_t29/10000;
spikeT0_t29=time2_t29;
[procInd0_t29,procInd1_t29]=histc(spikeT0_t29,tlin);
procInd_t29=find(procInd0_t29);
spikeT_t29=tlin(procInd_t29);
spike_t29=procInd0_t29';
[~,rawInd0_t29]=histc(spikeT0_t29,time2_t29);
markAll_t29(:,1)=procInd1_t29;markAll_t29(:,2:5)=mark0_t29(rawInd0_t29(rawInd0_t29~=0),:);
ms=min(min(markAll_t29(:,2:5))):mdel:max(max(markAll_t29(:,2:5)));
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum_t29=normpdf(xs'*ones(1,length(spikeT0_t29)),ones(length(xs),1)*xtrain(procInd1_t29),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t29=sum(Xnum_t29,2)./occ(:,1)./dt; %integral
Lint_t29=Lint_t29./sum(Lint_t29);


%% bookkeeping code: which spike comes which tetrode
time0=[time_t1;time_t2;time_t4;time_t5;time_t7;time_t10;time_t11;time_t12;time_t13;time_t14;time_t17;time_t18;time_t19;time_t20;time_t22;time_t23;time_t27;time_t29];
[time,timeInd]=sort(time0);
mark0=[mark0_t1;mark0_t2;mark0_t4;mark0_t5;mark0_t7;mark0_t10;mark0_t11;mark0_t12;mark0_t13;mark0_t14;mark0_t17;mark0_t18;mark0_t19;mark0_t20;mark0_t22;mark0_t23;mark0_t27;mark0_t29];
mark0=mark0(timeInd,:);
procInd1=[procInd1_t1;procInd1_t2;procInd1_t4;procInd1_t5;procInd1_t7;procInd1_t10;procInd1_t11;procInd1_t12;procInd1_t13;procInd1_t14;procInd1_t17;procInd1_t18;procInd1_t19;procInd1_t20;procInd1_t22;procInd1_t23;procInd1_t27;procInd1_t29];
procInd1=procInd1(timeInd,:);

len_t1=length(time_t1);len_t2=length(time_t2);len_t4=length(time_t4);len_t5=length(time_t5);
len_t7=length(time_t7);len_t11=length(time_t11);len_t10=length(time_t10);len_t12=length(time_t12);
len_t13=length(time_t13);len_t14=length(time_t14);len_t17=length(time_t17);len_t18=length(time_t18);
len_t19=length(time_t19);len_t20=length(time_t20);len_t22=length(time_t22);len_t23=length(time_t23);
len_t27=length(time_t27);len_t29=length(time_t29);

tet_ind=zeros(length(time),5);
%indicator matrix
%row: time point; column: which tetrode spikes
for i=1:length(time0)
    if timeInd(i)>=1 && timeInd(i)<=len_t1
        tet_ind(i,1)=1; %tet1
    elseif timeInd(i)>=len_t1+1 && timeInd(i)<=len_t1+len_t2
        tet_ind(i,2)=1; %tet2
    elseif timeInd(i)>=len_t1+len_t2+1 && timeInd(i)<=len_t1+len_t2+len_t4
        tet_ind(i,3)=1; %tet4
    elseif timeInd(i)>=len_t1+len_t2+len_t4+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5
        tet_ind(i,4)=1; %tet5
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7
        tet_ind(i,5)=1; %tet7
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10
        tet_ind(i,6)=1; %tet10
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11
        tet_ind(i,7)=1; %tet11
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12
        tet_ind(i,8)=1; %tet12
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13
        tet_ind(i,9)=1; %tet13
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14
        tet_ind(i,10)=1; %tet14
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17
        tet_ind(i,11)=1; %tet17
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18
        tet_ind(i,12)=1; %tet18
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19
        tet_ind(i,13)=1; %tet19
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20
        tet_ind(i,14)=1; %tet20
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22
        tet_ind(i,15)=1; %tet22
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22+len_t23
        tet_ind(i,16)=1; %tet23
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22+len_t23+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22+len_t23+len_t27
        tet_ind(i,17)=1; %tet27
    elseif timeInd(i)>=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22+len_t23+len_t27+1 && timeInd(i)<=len_t1+len_t2+len_t4+len_t5+len_t7+len_t10+len_t11+len_t12+len_t13+len_t14+len_t17+len_t18+len_t19+len_t20+len_t22+len_t23+len_t27+len_t29
        tet_ind(i,18)=1; %tet29
    end
end

tet_sum=tet_ind.*cumsum(tet_ind,1); %row: time point; column: index of spike per tetrode

%% caculate captial LAMBDA (when there is no spike on any tetrode)
ms=min(mark0(:)):mdel:max(mark0(:)); 
occ=normpdf(xs'*ones(1,T),ones(length(xs),1)*xtrain,sxker)*ones(T,length(ms));
%occ: columns are identical; occupancy based on position; denominator
Xnum=normpdf(xs'*ones(1,length(time0)),ones(length(xs),1)*xtrain(procInd1),sxker);
%Xnum: Gaussian kernel estimators for position
Lint=sum(Xnum,2)./occ(:,1)./dt; %integral
Lint=Lint./sum(Lint);
%Lint: conditional intensity function for the unmarked case



%% captial LAMBDA conditioned on I=1 and I=0
procInd1_I_out=procInd1(ismember(procInd1,ind_I_out));
occ_I_out=normpdf(xs'*ones(1,length(ind_I_out)),ones(length(xs),1)*xtrain(ind_I_out),sxker)*ones(length(ind_I_out),length(ms));
Xnum_I_out=normpdf(xs'*ones(1,length(xtrain(procInd1_I_out))),ones(length(xs),1)*xtrain(procInd1_I_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_I_out=sum(Xnum_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_I_out=Lint_I_out./sum(Lint_I_out);


procInd1_I_in=procInd1(ismember(procInd1,ind_I_in));
occ_I_in=normpdf(xs'*ones(1,length(ind_I_in)),ones(length(xs),1)*xtrain(ind_I_in),sxker)*ones(length(ind_I_in),length(ms));
Xnum_I_in=normpdf(xs'*ones(1,length(xtrain(procInd1_I_in))),ones(length(xs),1)*xtrain(procInd1_I_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_I_in=sum(Xnum_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_I_in=Lint_I_in./sum(Lint_I_in);




%% encode per tetrode, conditioning on I=1 and I=0
procInd1_t1_Ia_out=procInd1_t1(ismember(procInd1_t1,ind_I_out));
procInd1_t1_I_out=find(ismember(procInd1_t1,ind_I_out));
Xnum_t1_I_out=normpdf(xs'*ones(1,length(procInd1_t1_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t1_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t1_I_out=sum(Xnum_t1_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t1_I_out=Lint_t1_I_out./sum(Lint_t1_I_out);
procInd1_t1_Ia_in=procInd1_t1(ismember(procInd1_t1,ind_I_in));
procInd1_t1_I_in=find(ismember(procInd1_t1,ind_I_in));
Xnum_t1_I_in=normpdf(xs'*ones(1,length(procInd1_t1_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t1_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t1_I_in=sum(Xnum_t1_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t1_I_in=Lint_t1_I_in./sum(Lint_t1_I_in);

procInd1_t2_Ia_out=procInd1_t2(ismember(procInd1_t2,ind_I_out));
procInd1_t2_I_out=find(ismember(procInd1_t2,ind_I_out));
Xnum_t2_I_out=normpdf(xs'*ones(1,length(procInd1_t2_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t2_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t2_I_out=sum(Xnum_t2_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t2_I_out=Lint_t2_I_out./sum(Lint_t2_I_out);
procInd1_t2_Ia_in=procInd1_t2(ismember(procInd1_t2,ind_I_in));
procInd1_t2_I_in=find(ismember(procInd1_t2,ind_I_in));
Xnum_t2_I_in=normpdf(xs'*ones(1,length(procInd1_t2_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t2_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t2_I_in=sum(Xnum_t2_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t2_I_in=Lint_t2_I_in./sum(Lint_t2_I_in);

procInd1_t4_Ia_out=procInd1_t4(ismember(procInd1_t4,ind_I_out));
procInd1_t4_I_out=find(ismember(procInd1_t4,ind_I_out));
Xnum_t4_I_out=normpdf(xs'*ones(1,length(procInd1_t4_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t4_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t4_I_out=sum(Xnum_t4_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t4_I_out=Lint_t4_I_out./sum(Lint_t4_I_out);
procInd1_t4_Ia_in=procInd1_t4(ismember(procInd1_t4,ind_I_in));
procInd1_t4_I_in=find(ismember(procInd1_t4,ind_I_in));
Xnum_t4_I_in=normpdf(xs'*ones(1,length(procInd1_t4_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t4_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t4_I_in=sum(Xnum_t4_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t4_I_in=Lint_t4_I_in./sum(Lint_t4_I_in);

procInd1_t5_Ia_out=procInd1_t5(ismember(procInd1_t5,ind_I_out));
procInd1_t5_I_out=find(ismember(procInd1_t5,ind_I_out));
Xnum_t5_I_out=normpdf(xs'*ones(1,length(procInd1_t5_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t5_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t5_I_out=sum(Xnum_t5_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t5_I_out=Lint_t5_I_out./sum(Lint_t5_I_out);
procInd1_t5_Ia_in=procInd1_t5(ismember(procInd1_t5,ind_I_in));
procInd1_t5_I_in=find(ismember(procInd1_t5,ind_I_in));
Xnum_t5_I_in=normpdf(xs'*ones(1,length(procInd1_t5_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t5_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t5_I_in=sum(Xnum_t5_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t5_I_in=Lint_t5_I_in./sum(Lint_t5_I_in);

procInd1_t7_Ia_out=procInd1_t7(ismember(procInd1_t7,ind_I_out));
procInd1_t7_I_out=find(ismember(procInd1_t7,ind_I_out));
Xnum_t7_I_out=normpdf(xs'*ones(1,length(procInd1_t7_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t7_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t7_I_out=sum(Xnum_t7_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t7_I_out=Lint_t7_I_out./sum(Lint_t7_I_out);
procInd1_t7_Ia_in=procInd1_t7(ismember(procInd1_t7,ind_I_in));
procInd1_t7_I_in=find(ismember(procInd1_t7,ind_I_in));
Xnum_t7_I_in=normpdf(xs'*ones(1,length(procInd1_t7_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t7_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t7_I_in=sum(Xnum_t7_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t7_I_in=Lint_t7_I_in./sum(Lint_t7_I_in);

procInd1_t10_Ia_out=procInd1_t10(ismember(procInd1_t10,ind_I_out));
procInd1_t10_I_out=find(ismember(procInd1_t10,ind_I_out));
Xnum_t10_I_out=normpdf(xs'*ones(1,length(procInd1_t10_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t10_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t10_I_out=sum(Xnum_t10_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t10_I_out=Lint_t10_I_out./sum(Lint_t10_I_out);
procInd1_t10_Ia_in=procInd1_t10(ismember(procInd1_t10,ind_I_in));
procInd1_t10_I_in=find(ismember(procInd1_t10,ind_I_in));
Xnum_t10_I_in=normpdf(xs'*ones(1,length(procInd1_t10_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t10_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t10_I_in=sum(Xnum_t10_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t10_I_in=Lint_t10_I_in./sum(Lint_t10_I_in);

procInd1_t11_Ia_out=procInd1_t11(ismember(procInd1_t11,ind_I_out));
procInd1_t11_I_out=find(ismember(procInd1_t11,ind_I_out));
Xnum_t11_I_out=normpdf(xs'*ones(1,length(procInd1_t11_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t11_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t11_I_out=sum(Xnum_t11_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t11_I_out=Lint_t11_I_out./sum(Lint_t11_I_out);
procInd1_t11_Ia_in=procInd1_t11(ismember(procInd1_t11,ind_I_in));
procInd1_t11_I_in=find(ismember(procInd1_t11,ind_I_in));
Xnum_t11_I_in=normpdf(xs'*ones(1,length(procInd1_t11_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t11_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t11_I_in=sum(Xnum_t11_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t11_I_in=Lint_t11_I_in./sum(Lint_t11_I_in);

procInd1_t12_Ia_out=procInd1_t12(ismember(procInd1_t12,ind_I_out));
procInd1_t12_I_out=find(ismember(procInd1_t12,ind_I_out));
Xnum_t12_I_out=normpdf(xs'*ones(1,length(procInd1_t12_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t12_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t12_I_out=sum(Xnum_t12_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t12_I_out=Lint_t12_I_out./sum(Lint_t12_I_out);
procInd1_t12_Ia_in=procInd1_t12(ismember(procInd1_t12,ind_I_in));
procInd1_t12_I_in=find(ismember(procInd1_t12,ind_I_in));
Xnum_t12_I_in=normpdf(xs'*ones(1,length(procInd1_t12_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t12_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t12_I_in=sum(Xnum_t12_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t12_I_in=Lint_t12_I_in./sum(Lint_t12_I_in);

procInd1_t13_Ia_out=procInd1_t13(ismember(procInd1_t13,ind_I_out));
procInd1_t13_I_out=find(ismember(procInd1_t13,ind_I_out));
Xnum_t13_I_out=normpdf(xs'*ones(1,length(procInd1_t13_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t13_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t13_I_out=sum(Xnum_t13_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t13_I_out=Lint_t13_I_out./sum(Lint_t13_I_out);
procInd1_t13_Ia_in=procInd1_t13(ismember(procInd1_t13,ind_I_in));
procInd1_t13_I_in=find(ismember(procInd1_t13,ind_I_in));
Xnum_t13_I_in=normpdf(xs'*ones(1,length(procInd1_t13_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t13_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t13_I_in=sum(Xnum_t13_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t13_I_in=Lint_t13_I_in./sum(Lint_t13_I_in);

procInd1_t14_Ia_out=procInd1_t14(ismember(procInd1_t14,ind_I_out));
procInd1_t14_I_out=find(ismember(procInd1_t14,ind_I_out));
Xnum_t14_I_out=normpdf(xs'*ones(1,length(procInd1_t14_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t14_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t14_I_out=sum(Xnum_t14_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t14_I_out=Lint_t14_I_out./sum(Lint_t14_I_out);
procInd1_t14_Ia_in=procInd1_t14(ismember(procInd1_t14,ind_I_in));
procInd1_t14_I_in=find(ismember(procInd1_t14,ind_I_in));
Xnum_t14_I_in=normpdf(xs'*ones(1,length(procInd1_t14_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t14_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t14_I_in=sum(Xnum_t14_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t14_I_in=Lint_t14_I_in./sum(Lint_t14_I_in);

procInd1_t17_Ia_out=procInd1_t17(ismember(procInd1_t17,ind_I_out));
procInd1_t17_I_out=find(ismember(procInd1_t17,ind_I_out));
Xnum_t17_I_out=normpdf(xs'*ones(1,length(procInd1_t17_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t17_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t17_I_out=sum(Xnum_t17_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t17_I_out=Lint_t17_I_out./sum(Lint_t17_I_out);
procInd1_t17_Ia_in=procInd1_t17(ismember(procInd1_t17,ind_I_in));
procInd1_t17_I_in=find(ismember(procInd1_t17,ind_I_in));
Xnum_t17_I_in=normpdf(xs'*ones(1,length(procInd1_t17_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t17_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t17_I_in=sum(Xnum_t17_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t17_I_in=Lint_t17_I_in./sum(Lint_t17_I_in);

procInd1_t18_Ia_out=procInd1_t18(ismember(procInd1_t18,ind_I_out));
procInd1_t18_I_out=find(ismember(procInd1_t18,ind_I_out));
Xnum_t18_I_out=normpdf(xs'*ones(1,length(procInd1_t18_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t18_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t18_I_out=sum(Xnum_t18_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t18_I_out=Lint_t18_I_out./sum(Lint_t18_I_out);
procInd1_t18_Ia_in=procInd1_t18(ismember(procInd1_t18,ind_I_in));
procInd1_t18_I_in=find(ismember(procInd1_t18,ind_I_in));
Xnum_t18_I_in=normpdf(xs'*ones(1,length(procInd1_t18_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t18_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t18_I_in=sum(Xnum_t18_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t18_I_in=Lint_t18_I_in./sum(Lint_t18_I_in);

procInd1_t19_Ia_out=procInd1_t19(ismember(procInd1_t19,ind_I_out));
procInd1_t19_I_out=find(ismember(procInd1_t19,ind_I_out));
Xnum_t19_I_out=normpdf(xs'*ones(1,length(procInd1_t19_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t19_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t19_I_out=sum(Xnum_t19_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t19_I_out=Lint_t19_I_out./sum(Lint_t19_I_out);
procInd1_t19_Ia_in=procInd1_t19(ismember(procInd1_t19,ind_I_out));
procInd1_t19_I_in=find(ismember(procInd1_t19,ind_I_out));
Xnum_t19_I_in=normpdf(xs'*ones(1,length(procInd1_t19_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t19_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t19_I_in=sum(Xnum_t19_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t19_I_in=Lint_t19_I_in./sum(Lint_t19_I_in);

procInd1_t20_Ia_out=procInd1_t20(ismember(procInd1_t20,ind_I_out));
procInd1_t20_I_out=find(ismember(procInd1_t20,ind_I_out));
Xnum_t20_I_out=normpdf(xs'*ones(1,length(procInd1_t20_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t20_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t20_I_out=sum(Xnum_t20_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t20_I_out=Lint_t20_I_out./sum(Lint_t20_I_out);
procInd1_t20_Ia_in=procInd1_t20(ismember(procInd1_t20,ind_I_in));
procInd1_t20_I_in=find(ismember(procInd1_t20,ind_I_in));
Xnum_t20_I_in=normpdf(xs'*ones(1,length(procInd1_t20_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t20_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t20_I_in=sum(Xnum_t20_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t20_I_in=Lint_t20_I_in./sum(Lint_t20_I_in);

procInd1_t22_Ia_out=procInd1_t22(ismember(procInd1_t22,ind_I_out));
procInd1_t22_I_out=find(ismember(procInd1_t22,ind_I_out));
Xnum_t22_I_out=normpdf(xs'*ones(1,length(procInd1_t22_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t22_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t22_I_out=sum(Xnum_t22_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t22_I_out=Lint_t22_I_out./sum(Lint_t22_I_out);
procInd1_t22_Ia_in=procInd1_t22(ismember(procInd1_t22,ind_I_out));
procInd1_t22_I_in=find(ismember(procInd1_t22,ind_I_out));
Xnum_t22_I_in=normpdf(xs'*ones(1,length(procInd1_t22_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t22_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t22_I_in=sum(Xnum_t22_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t22_I_in=Lint_t22_I_in./sum(Lint_t22_I_in);

procInd1_t23_Ia_out=procInd1_t23(ismember(procInd1_t23,ind_I_out));
procInd1_t23_I_out=find(ismember(procInd1_t23,ind_I_out));
Xnum_t23_I_out=normpdf(xs'*ones(1,length(procInd1_t23_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t23_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t23_I_out=sum(Xnum_t23_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t23_I_out=Lint_t23_I_out./sum(Lint_t23_I_out);
procInd1_t23_Ia_in=procInd1_t23(ismember(procInd1_t23,ind_I_in));
procInd1_t23_I_in=find(ismember(procInd1_t23,ind_I_in));
Xnum_t23_I_in=normpdf(xs'*ones(1,length(procInd1_t23_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t23_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t23_I_in=sum(Xnum_t23_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t23_I_in=Lint_t23_I_in./sum(Lint_t23_I_in);

procInd1_t27_Ia_out=procInd1_t27(ismember(procInd1_t27,ind_I_out));
procInd1_t27_I_out=find(ismember(procInd1_t27,ind_I_out));
Xnum_t27_I_out=normpdf(xs'*ones(1,length(procInd1_t27_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t27_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t27_I_out=sum(Xnum_t27_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t27_I_out=Lint_t27_I_out./sum(Lint_t27_I_out);
procInd1_t27_Ia_in=procInd1_t27(ismember(procInd1_t27,ind_I_in));
procInd1_t27_I_in=find(ismember(procInd1_t27,ind_I_in));
Xnum_t27_I_in=normpdf(xs'*ones(1,length(procInd1_t27_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t27_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t27_I_in=sum(Xnum_t27_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t27_I_in=Lint_t27_I_in./sum(Lint_t27_I_in);

procInd1_t29_Ia_out=procInd1_t29(ismember(procInd1_t29,ind_I_out));
procInd1_t29_I_out=find(ismember(procInd1_t29,ind_I_out));
Xnum_t29_I_out=normpdf(xs'*ones(1,length(procInd1_t29_Ia_out)),ones(length(xs),1)*xtrain(procInd1_t29_Ia_out),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t29_I_out=sum(Xnum_t29_I_out,2)./occ_I_out(:,1)./dt; %integral
Lint_t29_I_out=Lint_t29_I_out./sum(Lint_t29_I_out);
procInd1_t29_Ia_in=procInd1_t29(ismember(procInd1_t29,ind_I_in));
procInd1_t29_I_in=find(ismember(procInd1_t29,ind_I_in));
Xnum_t29_I_in=normpdf(xs'*ones(1,length(procInd1_t29_Ia_in)),ones(length(xs),1)*xtrain(procInd1_t29_Ia_in),sxker);
%Xnum: Gaussian kernel estimators for position
Lint_t29_I_in=sum(Xnum_t29_I_in,2)./occ_I_in(:,1)./dt; %integral
Lint_t29_I_in=Lint_t29_I_in./sum(Lint_t29_I_in);




%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%     DECODE        %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
velocity=pos{ex}{ep}.data(:,5);
%linVel=linpos{ex}{ep}.statematrix.linearVelocity;
clear rloc vel;
for pic=1:length(rippleI)
    rIndV=pic; %5, 12
    rloc_Ind=find(A*1000>ti(ripple_seg(rippleI(rIndV),1))&A*1000<ti(ripple_seg(rippleI(rIndV),2)));

    rloc(pic)=vecLF(rloc_Ind(1),2);
    vel(pic)=velocity(rloc_Ind(1),1);
end

Ind_vel=find(vel<4);length(Ind_vel) 
%only decode replay when the running speed < 4cm/sec
ripplesconsN=traj_Ind(rippleI(Ind_vel));

%% decoder
clear sumStat;
for pic=1:length(Ind_vel)
rIndV=Ind_vel(pic); %5, 12

spike_tim=ripple_seg(rippleI(rIndV),1):ripple_seg(rippleI(rIndV),2); %from 1 to 90000~
numSteps=length(spike_tim);
xi=round(time/10);

%%
dt=1/33.4;spike_r=zeros(18,numSteps);
n=length(stateV);
numSteps=size(spike_r,2);
%P(x0|I);
Px_I_out=exp(-stateV.^2./(2*(2*xdel)^2));Px_I_out=Px_I_out./sum(Px_I_out);
Px_I_in=max(Px_I_out)*ones(1,n)-Px_I_out; Px_I_in=Px_I_in./sum(Px_I_in);
Px_I0=Px_I_out;Px_I1=Px_I_in;Px_I2=Px_I_in;Px_I3=Px_I_out;
%P(x0)=P(x0|I)P(I);
postx_I0=0.25*Px_I_out';postx_I1=0.25*Px_I_in';postx_I2=0.25*Px_I_in';postx_I3=0.25*Px_I_out';
pI0_vec=zeros(numSteps,1);pI1_vec=zeros(numSteps,1);pI2_vec=zeros(numSteps,1);pI3_vec=zeros(numSteps,1);
postxM_r_I0=zeros(n,numSteps);postxM_r_I1=zeros(n,numSteps);postxM_r_I2=zeros(n,numSteps);postxM_r_I3=zeros(n,numSteps);
%state transition
stateM_I_out=stateM_I1_gausnorm;stateM_I_in=stateM_I0_gausnorm;
stateM_I0=stateM_I_out;stateM_I1=stateM_I_in;stateM_I2=stateM_I_in;stateM_I3=stateM_I_out;
for t=1:numSteps
    tt=spike_tim(t);
    aa=find(xi==ti(tt));
    
    onestep_I0=stateM_I0*postx_I0;
    onestep_I1=stateM_I1*postx_I1;
    onestep_I2=stateM_I2*postx_I2;
    onestep_I3=stateM_I3*postx_I3;    
    
    L_I0=ones(n,1);L_I1=ones(n,1);L_I2=ones(n,1);L_I3=ones(n,1);

    if isempty(aa)==1 %if no spike occurs at time t
        L_I0=exp(-Lint_I_out.*dt);L_I1=exp(-Lint_I_out.*dt);
        L_I2=exp(-Lint_I_in.*dt);L_I3=exp(-Lint_I_in.*dt);
        
    elseif isempty(aa)==0 %if spikes

        l_out=zeros(n,length(aa));
        for j=1:length(aa)
            jj=aa(j);
            tetVec=tet_ind(jj,:);
            
            if tetVec(1)==1 %tet1
                spike_r(1,t)=1;
                i=tet_sum(jj,1);
                l0=normpdf(markAll_t1(i,2)*ones(1,length(procInd1_t1_Ia_out)),markAll_t1(procInd1_t1_I_out,2)',smker).*normpdf(markAll_t1(i,3)*ones(1,length(procInd1_t1_Ia_out)),markAll_t1(procInd1_t1_I_out,3)',smker).*normpdf(markAll_t1(i,4)*ones(1,length(procInd1_t1_Ia_out)),markAll_t1(procInd1_t1_I_out,4)',smker).*normpdf(markAll_t1(i,5)*ones(1,length(procInd1_t1_Ia_out)),markAll_t1(procInd1_t1_I_out,5)',smker);
                l1=Xnum_t1_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t1_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(2)==1 %tet2
                spike_r(2,t)=1;
                i=tet_sum(jj,2);
                l0=normpdf(markAll_t2(i,2)*ones(1,length(procInd1_t2_Ia_out)),markAll_t2(procInd1_t2_I_out,2)',smker).*normpdf(markAll_t2(i,3)*ones(1,length(procInd1_t2_Ia_out)),markAll_t2(procInd1_t2_I_out,3)',smker).*normpdf(markAll_t2(i,4)*ones(1,length(procInd1_t2_Ia_out)),markAll_t2(procInd1_t2_I_out,4)',smker).*normpdf(markAll_t2(i,5)*ones(1,length(procInd1_t2_Ia_out)),markAll_t2(procInd1_t2_I_out,5)',smker);
                l1=Xnum_t2_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t2_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(3)==1 %tet4
                spike_r(3,t)=1;
                i=tet_sum(jj,3);
                l0=normpdf(markAll_t4(i,2)*ones(1,length(procInd1_t4_Ia_out)),markAll_t4(procInd1_t4_I_out,2)',smker).*normpdf(markAll_t4(i,3)*ones(1,length(procInd1_t4_Ia_out)),markAll_t4(procInd1_t4_I_out,3)',smker).*normpdf(markAll_t4(i,4)*ones(1,length(procInd1_t4_Ia_out)),markAll_t4(procInd1_t4_I_out,4)',smker).*normpdf(markAll_t4(i,5)*ones(1,length(procInd1_t4_Ia_out)),markAll_t4(procInd1_t4_I_out,5)',smker);
                l1=Xnum_t4_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t4_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(4)==1 %tet5
                spike_r(4,t)=1;
                i=tet_sum(jj,4);
                l0=normpdf(markAll_t5(i,2)*ones(1,length(procInd1_t5_Ia_out)),markAll_t5(procInd1_t5_I_out,2)',smker).*normpdf(markAll_t5(i,3)*ones(1,length(procInd1_t5_Ia_out)),markAll_t5(procInd1_t5_I_out,3)',smker).*normpdf(markAll_t5(i,4)*ones(1,length(procInd1_t5_Ia_out)),markAll_t5(procInd1_t5_I_out,4)',smker).*normpdf(markAll_t5(i,5)*ones(1,length(procInd1_t5_Ia_out)),markAll_t5(procInd1_t5_I_out,5)',smker);
                l1=Xnum_t5_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t5_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(5)==1 %tet7
                spike_r(5,t)=1;
                i=tet_sum(jj,5);
                l0=normpdf(markAll_t7(i,2)*ones(1,length(procInd1_t7_Ia_out)),markAll_t7(procInd1_t7_I_out,2)',smker).*normpdf(markAll_t7(i,3)*ones(1,length(procInd1_t7_Ia_out)),markAll_t7(procInd1_t7_I_out,3)',smker).*normpdf(markAll_t7(i,4)*ones(1,length(procInd1_t7_Ia_out)),markAll_t7(procInd1_t7_I_out,4)',smker).*normpdf(markAll_t7(i,5)*ones(1,length(procInd1_t7_Ia_out)),markAll_t7(procInd1_t7_I_out,5)',smker);
                l1=Xnum_t7_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t7_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(6)==1 %tet10
                spike_r(6,t)=1;
                i=tet_sum(jj,6);
                l0=normpdf(markAll_t10(i,2)*ones(1,length(procInd1_t10_Ia_out)),markAll_t10(procInd1_t10_I_out,2)',smker).*normpdf(markAll_t10(i,3)*ones(1,length(procInd1_t10_Ia_out)),markAll_t10(procInd1_t10_I_out,3)',smker).*normpdf(markAll_t10(i,4)*ones(1,length(procInd1_t10_Ia_out)),markAll_t10(procInd1_t10_I_out,4)',smker).*normpdf(markAll_t10(i,5)*ones(1,length(procInd1_t10_Ia_out)),markAll_t10(procInd1_t10_I_out,5)',smker);
                l1=Xnum_t10_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t10_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(7)==1 %tet11
                spike_r(7,t)=1;
                i=tet_sum(jj,7);
                l0=normpdf(markAll_t11(i,2)*ones(1,length(procInd1_t11_Ia_out)),markAll_t11(procInd1_t11_I_out,2)',smker).*normpdf(markAll_t11(i,3)*ones(1,length(procInd1_t11_Ia_out)),markAll_t11(procInd1_t11_I_out,3)',smker).*normpdf(markAll_t11(i,4)*ones(1,length(procInd1_t11_Ia_out)),markAll_t11(procInd1_t11_I_out,4)',smker).*normpdf(markAll_t11(i,5)*ones(1,length(procInd1_t11_Ia_out)),markAll_t11(procInd1_t11_I_out,5)',smker);
                l1=Xnum_t11_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t11_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(8)==1 %tet12
                spike_r(8,t)=1;
                i=tet_sum(jj,8);
                l0=normpdf(markAll_t12(i,2)*ones(1,length(procInd1_t12_Ia_out)),markAll_t12(procInd1_t12_I_out,2)',smker).*normpdf(markAll_t12(i,3)*ones(1,length(procInd1_t12_Ia_out)),markAll_t12(procInd1_t12_I_out,3)',smker).*normpdf(markAll_t12(i,4)*ones(1,length(procInd1_t12_Ia_out)),markAll_t12(procInd1_t12_I_out,4)',smker).*normpdf(markAll_t12(i,5)*ones(1,length(procInd1_t12_Ia_out)),markAll_t12(procInd1_t12_I_out,5)',smker);
                l1=Xnum_t12_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t12_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(9)==1 %tet13
                spike_r(9,t)=1;
                i=tet_sum(jj,9);
                l0=normpdf(markAll_t13(i,2)*ones(1,length(procInd1_t13_Ia_out)),markAll_t13(procInd1_t13_I_out,2)',smker).*normpdf(markAll_t13(i,3)*ones(1,length(procInd1_t13_Ia_out)),markAll_t13(procInd1_t13_I_out,3)',smker).*normpdf(markAll_t13(i,4)*ones(1,length(procInd1_t13_Ia_out)),markAll_t13(procInd1_t13_I_out,4)',smker).*normpdf(markAll_t13(i,5)*ones(1,length(procInd1_t13_Ia_out)),markAll_t13(procInd1_t13_I_out,5)',smker);
                l1=Xnum_t13_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t13_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(10)==1 %tet14
                spike_r(10,t)=1;
                i=tet_sum(jj,10);
                l0=normpdf(markAll_t14(i,2)*ones(1,length(procInd1_t14_Ia_out)),markAll_t14(procInd1_t14_I_out,2)',smker).*normpdf(markAll_t14(i,3)*ones(1,length(procInd1_t14_Ia_out)),markAll_t14(procInd1_t14_I_out,3)',smker).*normpdf(markAll_t14(i,4)*ones(1,length(procInd1_t14_Ia_out)),markAll_t14(procInd1_t14_I_out,4)',smker).*normpdf(markAll_t14(i,5)*ones(1,length(procInd1_t14_Ia_out)),markAll_t14(procInd1_t14_I_out,5)',smker);
                l1=Xnum_t14_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t14_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(11)==1 %tet17
                spike_r(11,t)=1;
                i=tet_sum(jj,11);
                l0=normpdf(markAll_t17(i,2)*ones(1,length(procInd1_t17_Ia_out)),markAll_t17(procInd1_t17_I_out,2)',smker).*normpdf(markAll_t17(i,3)*ones(1,length(procInd1_t17_Ia_out)),markAll_t17(procInd1_t17_I_out,3)',smker).*normpdf(markAll_t17(i,4)*ones(1,length(procInd1_t17_Ia_out)),markAll_t17(procInd1_t17_I_out,4)',smker).*normpdf(markAll_t17(i,5)*ones(1,length(procInd1_t17_Ia_out)),markAll_t17(procInd1_t17_I_out,5)',smker);
                l1=Xnum_t17_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t17_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(12)==1 %tet18
                spike_r(12,t)=1;
                i=tet_sum(jj,12);
                l0=normpdf(markAll_t18(i,2)*ones(1,length(procInd1_t18_Ia_out)),markAll_t18(procInd1_t18_I_out,2)',smker).*normpdf(markAll_t18(i,3)*ones(1,length(procInd1_t18_Ia_out)),markAll_t18(procInd1_t18_I_out,3)',smker).*normpdf(markAll_t18(i,4)*ones(1,length(procInd1_t18_Ia_out)),markAll_t18(procInd1_t18_I_out,4)',smker).*normpdf(markAll_t18(i,5)*ones(1,length(procInd1_t18_Ia_out)),markAll_t18(procInd1_t18_I_out,5)',smker);
                l1=Xnum_t18_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t18_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(13)==1 %tet19
                spike_r(13,t)=1;
                i=tet_sum(jj,13);
                l0=normpdf(markAll_t19(i,2)*ones(1,length(procInd1_t19_Ia_out)),markAll_t19(procInd1_t19_I_out,2)',smker).*normpdf(markAll_t19(i,3)*ones(1,length(procInd1_t19_Ia_out)),markAll_t19(procInd1_t19_I_out,3)',smker).*normpdf(markAll_t19(i,4)*ones(1,length(procInd1_t19_Ia_out)),markAll_t19(procInd1_t19_I_out,4)',smker).*normpdf(markAll_t19(i,5)*ones(1,length(procInd1_t19_Ia_out)),markAll_t19(procInd1_t19_I_out,5)',smker);
                l1=Xnum_t19_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t19_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(14)==1 %tet20
                spike_r(14,t)=1;
                i=tet_sum(jj,14);
                l0=normpdf(markAll_t20(i,2)*ones(1,length(procInd1_t20_Ia_out)),markAll_t20(procInd1_t20_I_out,2)',smker).*normpdf(markAll_t20(i,3)*ones(1,length(procInd1_t20_Ia_out)),markAll_t20(procInd1_t20_I_out,3)',smker).*normpdf(markAll_t20(i,4)*ones(1,length(procInd1_t20_Ia_out)),markAll_t20(procInd1_t20_I_out,4)',smker).*normpdf(markAll_t20(i,5)*ones(1,length(procInd1_t20_Ia_out)),markAll_t20(procInd1_t20_I_out,5)',smker);
                l1=Xnum_t20_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t20_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(15)==1 %tet22
                spike_r(15,t)=1;
                i=tet_sum(jj,15);
                l0=normpdf(markAll_t22(i,2)*ones(1,length(procInd1_t22_Ia_out)),markAll_t22(procInd1_t22_I_out,2)',smker).*normpdf(markAll_t22(i,3)*ones(1,length(procInd1_t22_Ia_out)),markAll_t22(procInd1_t22_I_out,3)',smker).*normpdf(markAll_t22(i,4)*ones(1,length(procInd1_t22_Ia_out)),markAll_t22(procInd1_t22_I_out,4)',smker).*normpdf(markAll_t22(i,5)*ones(1,length(procInd1_t22_Ia_out)),markAll_t22(procInd1_t22_I_out,5)',smker);
                l1=Xnum_t22_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t22_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(16)==1 %tet23
                spike_r(16,t)=1;
                i=tet_sum(jj,16);
                l0=normpdf(markAll_t23(i,2)*ones(1,length(procInd1_t23_Ia_out)),markAll_t23(procInd1_t23_I_out,2)',smker).*normpdf(markAll_t23(i,3)*ones(1,length(procInd1_t23_Ia_out)),markAll_t23(procInd1_t23_I_out,3)',smker).*normpdf(markAll_t23(i,4)*ones(1,length(procInd1_t23_Ia_out)),markAll_t23(procInd1_t23_I_out,4)',smker).*normpdf(markAll_t23(i,5)*ones(1,length(procInd1_t23_Ia_out)),markAll_t23(procInd1_t23_I_out,5)',smker);
                l1=Xnum_t23_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t23_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(17)==1 %tet27
                spike_r(17,t)=1;
                i=tet_sum(jj,17);
                l0=normpdf(markAll_t27(i,2)*ones(1,length(procInd1_t27_Ia_out)),markAll_t27(procInd1_t27_I_out,2)',smker).*normpdf(markAll_t27(i,3)*ones(1,length(procInd1_t27_Ia_out)),markAll_t27(procInd1_t27_I_out,3)',smker).*normpdf(markAll_t27(i,4)*ones(1,length(procInd1_t27_Ia_out)),markAll_t27(procInd1_t27_I_out,4)',smker).*normpdf(markAll_t27(i,5)*ones(1,length(procInd1_t27_Ia_out)),markAll_t27(procInd1_t27_I_out,5)',smker);
                l1=Xnum_t27_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t27_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            elseif tetVec(18)==1 %tet29
                spike_r(18,t)=1;
                i=tet_sum(jj,18);
                l0=normpdf(markAll_t29(i,2)*ones(1,length(procInd1_t29_Ia_out)),markAll_t29(procInd1_t29_I_out,2)',smker).*normpdf(markAll_t29(i,3)*ones(1,length(procInd1_t29_Ia_out)),markAll_t29(procInd1_t29_I_out,3)',smker).*normpdf(markAll_t29(i,4)*ones(1,length(procInd1_t29_Ia_out)),markAll_t29(procInd1_t29_I_out,4)',smker).*normpdf(markAll_t29(i,5)*ones(1,length(procInd1_t29_Ia_out)),markAll_t29(procInd1_t29_I_out,5)',smker);
                l1=Xnum_t29_I_out*l0'./occ_I_out(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t29_I_out.*dt);
                l2=l2./sum(l2);
                l_out(:,j)=l2;
            end
        end
        L_out=prod(l_out,2);L_out=L_out./sum(L_out);
                
        l_in=zeros(n,length(aa)); 
        for j=1:length(aa)
            jj=aa(j);
            tetVec=tet_ind(jj,:);
            
            if tetVec(1)==1 %tet1
                spike_r(1,t)=1;
                i=tet_sum(jj,1);
                l0=normpdf(markAll_t1(i,2)*ones(1,length(procInd1_t1_Ia_in)),markAll_t1(procInd1_t1_I_in,2)',smker).*normpdf(markAll_t1(i,3)*ones(1,length(procInd1_t1_Ia_in)),markAll_t1(procInd1_t1_I_in,3)',smker).*normpdf(markAll_t1(i,4)*ones(1,length(procInd1_t1_Ia_in)),markAll_t1(procInd1_t1_I_in,4)',smker).*normpdf(markAll_t1(i,5)*ones(1,length(procInd1_t1_Ia_in)),markAll_t1(procInd1_t1_I_in,5)',smker);
                l1=Xnum_t1_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t1_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(2)==1 %tet2
                spike_r(2,t)=1;
                i=tet_sum(jj,2);
                l0=normpdf(markAll_t2(i,2)*ones(1,length(procInd1_t2_Ia_in)),markAll_t2(procInd1_t2_I_in,2)',smker).*normpdf(markAll_t2(i,3)*ones(1,length(procInd1_t2_Ia_in)),markAll_t2(procInd1_t2_I_in,3)',smker).*normpdf(markAll_t2(i,4)*ones(1,length(procInd1_t2_Ia_in)),markAll_t2(procInd1_t2_I_in,4)',smker).*normpdf(markAll_t2(i,5)*ones(1,length(procInd1_t2_Ia_in)),markAll_t2(procInd1_t2_I_in,5)',smker);
                l1=Xnum_t2_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t2_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(3)==1 %tet4
                spike_r(3,t)=1;
                i=tet_sum(jj,3);
                l0=normpdf(markAll_t4(i,2)*ones(1,length(procInd1_t4_Ia_in)),markAll_t4(procInd1_t4_I_in,2)',smker).*normpdf(markAll_t4(i,3)*ones(1,length(procInd1_t4_Ia_in)),markAll_t4(procInd1_t4_I_in,3)',smker).*normpdf(markAll_t4(i,4)*ones(1,length(procInd1_t4_Ia_in)),markAll_t4(procInd1_t4_I_in,4)',smker).*normpdf(markAll_t4(i,5)*ones(1,length(procInd1_t4_Ia_in)),markAll_t4(procInd1_t4_I_in,5)',smker);
                l1=Xnum_t4_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t4_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(4)==1 %tet5
                spike_r(4,t)=1;
                i=tet_sum(jj,4);
                l0=normpdf(markAll_t5(i,2)*ones(1,length(procInd1_t5_Ia_in)),markAll_t5(procInd1_t5_I_in,2)',smker).*normpdf(markAll_t5(i,3)*ones(1,length(procInd1_t5_Ia_in)),markAll_t5(procInd1_t5_I_in,3)',smker).*normpdf(markAll_t5(i,4)*ones(1,length(procInd1_t5_Ia_in)),markAll_t5(procInd1_t5_I_in,4)',smker).*normpdf(markAll_t5(i,5)*ones(1,length(procInd1_t5_Ia_in)),markAll_t5(procInd1_t5_I_in,5)',smker);
                l1=Xnum_t5_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t5_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(5)==1 %tet7
                spike_r(5,t)=1;
                i=tet_sum(jj,5);
                l0=normpdf(markAll_t7(i,2)*ones(1,length(procInd1_t7_Ia_in)),markAll_t7(procInd1_t7_I_in,2)',smker).*normpdf(markAll_t7(i,3)*ones(1,length(procInd1_t7_Ia_in)),markAll_t7(procInd1_t7_I_in,3)',smker).*normpdf(markAll_t7(i,4)*ones(1,length(procInd1_t7_Ia_in)),markAll_t7(procInd1_t7_I_in,4)',smker).*normpdf(markAll_t7(i,5)*ones(1,length(procInd1_t7_Ia_in)),markAll_t7(procInd1_t7_I_in,5)',smker);
                l1=Xnum_t7_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t7_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(6)==1 %tet10
                spike_r(6,t)=1;
                i=tet_sum(jj,6);
                l0=normpdf(markAll_t10(i,2)*ones(1,length(procInd1_t10_Ia_in)),markAll_t10(procInd1_t10_I_in,2)',smker).*normpdf(markAll_t10(i,3)*ones(1,length(procInd1_t10_Ia_in)),markAll_t10(procInd1_t10_I_in,3)',smker).*normpdf(markAll_t10(i,4)*ones(1,length(procInd1_t10_Ia_in)),markAll_t10(procInd1_t10_I_in,4)',smker).*normpdf(markAll_t10(i,5)*ones(1,length(procInd1_t10_Ia_in)),markAll_t10(procInd1_t10_I_in,5)',smker);
                l1=Xnum_t10_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t10_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(7)==1 %tet11
                spike_r(7,t)=1;
                i=tet_sum(jj,7);
                l0=normpdf(markAll_t11(i,2)*ones(1,length(procInd1_t11_Ia_in)),markAll_t11(procInd1_t11_I_in,2)',smker).*normpdf(markAll_t11(i,3)*ones(1,length(procInd1_t11_Ia_in)),markAll_t11(procInd1_t11_I_in,3)',smker).*normpdf(markAll_t11(i,4)*ones(1,length(procInd1_t11_Ia_in)),markAll_t11(procInd1_t11_I_in,4)',smker).*normpdf(markAll_t11(i,5)*ones(1,length(procInd1_t11_Ia_in)),markAll_t11(procInd1_t11_I_in,5)',smker);
                l1=Xnum_t11_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t11_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(8)==1 %tet12
                spike_r(8,t)=1;
                i=tet_sum(jj,8);
                l0=normpdf(markAll_t12(i,2)*ones(1,length(procInd1_t12_Ia_in)),markAll_t12(procInd1_t12_I_in,2)',smker).*normpdf(markAll_t12(i,3)*ones(1,length(procInd1_t12_Ia_in)),markAll_t12(procInd1_t12_I_in,3)',smker).*normpdf(markAll_t12(i,4)*ones(1,length(procInd1_t12_Ia_in)),markAll_t12(procInd1_t12_I_in,4)',smker).*normpdf(markAll_t12(i,5)*ones(1,length(procInd1_t12_Ia_in)),markAll_t12(procInd1_t12_I_in,5)',smker);
                l1=Xnum_t12_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t12_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(9)==1 %tet13
                spike_r(9,t)=1;
                i=tet_sum(jj,9);
                l0=normpdf(markAll_t13(i,2)*ones(1,length(procInd1_t13_Ia_in)),markAll_t13(procInd1_t13_I_in,2)',smker).*normpdf(markAll_t13(i,3)*ones(1,length(procInd1_t13_Ia_in)),markAll_t13(procInd1_t13_I_in,3)',smker).*normpdf(markAll_t13(i,4)*ones(1,length(procInd1_t13_Ia_in)),markAll_t13(procInd1_t13_I_in,4)',smker).*normpdf(markAll_t13(i,5)*ones(1,length(procInd1_t13_Ia_in)),markAll_t13(procInd1_t13_I_in,5)',smker);
                l1=Xnum_t13_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t13_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(10)==1 %tet14
                spike_r(10,t)=1;
                i=tet_sum(jj,10);
                l0=normpdf(markAll_t14(i,2)*ones(1,length(procInd1_t14_Ia_in)),markAll_t14(procInd1_t14_I_in,2)',smker).*normpdf(markAll_t14(i,3)*ones(1,length(procInd1_t14_Ia_in)),markAll_t14(procInd1_t14_I_in,3)',smker).*normpdf(markAll_t14(i,4)*ones(1,length(procInd1_t14_Ia_in)),markAll_t14(procInd1_t14_I_in,4)',smker).*normpdf(markAll_t14(i,5)*ones(1,length(procInd1_t14_Ia_in)),markAll_t14(procInd1_t14_I_in,5)',smker);
                l1=Xnum_t14_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t14_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(11)==1 %tet17
                spike_r(11,t)=1;
                i=tet_sum(jj,11);
                l0=normpdf(markAll_t17(i,2)*ones(1,length(procInd1_t17_Ia_in)),markAll_t17(procInd1_t17_I_in,2)',smker).*normpdf(markAll_t17(i,3)*ones(1,length(procInd1_t17_Ia_in)),markAll_t17(procInd1_t17_I_in,3)',smker).*normpdf(markAll_t17(i,4)*ones(1,length(procInd1_t17_Ia_in)),markAll_t17(procInd1_t17_I_in,4)',smker).*normpdf(markAll_t17(i,5)*ones(1,length(procInd1_t17_Ia_in)),markAll_t17(procInd1_t17_I_in,5)',smker);
                l1=Xnum_t17_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t17_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(12)==1 %tet18
                spike_r(12,t)=1;
                i=tet_sum(jj,12);
                l0=normpdf(markAll_t18(i,2)*ones(1,length(procInd1_t18_Ia_in)),markAll_t18(procInd1_t18_I_in,2)',smker).*normpdf(markAll_t18(i,3)*ones(1,length(procInd1_t18_Ia_in)),markAll_t18(procInd1_t18_I_in,3)',smker).*normpdf(markAll_t18(i,4)*ones(1,length(procInd1_t18_Ia_in)),markAll_t18(procInd1_t18_I_in,4)',smker).*normpdf(markAll_t18(i,5)*ones(1,length(procInd1_t18_Ia_in)),markAll_t18(procInd1_t18_I_in,5)',smker);
                l1=Xnum_t18_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t18_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(13)==1 %tet19
                spike_r(13,t)=1;
                i=tet_sum(jj,13);
                l0=normpdf(markAll_t19(i,2)*ones(1,length(procInd1_t19_Ia_in)),markAll_t19(procInd1_t19_I_in,2)',smker).*normpdf(markAll_t19(i,3)*ones(1,length(procInd1_t19_Ia_in)),markAll_t19(procInd1_t19_I_in,3)',smker).*normpdf(markAll_t19(i,4)*ones(1,length(procInd1_t19_Ia_in)),markAll_t19(procInd1_t19_I_in,4)',smker).*normpdf(markAll_t19(i,5)*ones(1,length(procInd1_t19_Ia_in)),markAll_t19(procInd1_t19_I_in,5)',smker);
                l1=Xnum_t19_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t19_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(14)==1 %tet20
                spike_r(14,t)=1;
                i=tet_sum(jj,14);
                l0=normpdf(markAll_t20(i,2)*ones(1,length(procInd1_t20_Ia_in)),markAll_t20(procInd1_t20_I_in,2)',smker).*normpdf(markAll_t20(i,3)*ones(1,length(procInd1_t20_Ia_in)),markAll_t20(procInd1_t20_I_in,3)',smker).*normpdf(markAll_t20(i,4)*ones(1,length(procInd1_t20_Ia_in)),markAll_t20(procInd1_t20_I_in,4)',smker).*normpdf(markAll_t20(i,5)*ones(1,length(procInd1_t20_Ia_in)),markAll_t20(procInd1_t20_I_in,5)',smker);
                l1=Xnum_t20_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t20_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(15)==1 %tet22
                spike_r(15,t)=1;
                i=tet_sum(jj,15);
                l0=normpdf(markAll_t22(i,2)*ones(1,length(procInd1_t22_Ia_in)),markAll_t22(procInd1_t22_I_in,2)',smker).*normpdf(markAll_t22(i,3)*ones(1,length(procInd1_t22_Ia_in)),markAll_t22(procInd1_t22_I_in,3)',smker).*normpdf(markAll_t22(i,4)*ones(1,length(procInd1_t22_Ia_in)),markAll_t22(procInd1_t22_I_in,4)',smker).*normpdf(markAll_t22(i,5)*ones(1,length(procInd1_t22_Ia_in)),markAll_t22(procInd1_t22_I_in,5)',smker);
                l1=Xnum_t22_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t22_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(16)==1 %tet23
                spike_r(16,t)=1;
                i=tet_sum(jj,16);
                l0=normpdf(markAll_t23(i,2)*ones(1,length(procInd1_t23_Ia_in)),markAll_t23(procInd1_t23_I_in,2)',smker).*normpdf(markAll_t23(i,3)*ones(1,length(procInd1_t23_Ia_in)),markAll_t23(procInd1_t23_I_in,3)',smker).*normpdf(markAll_t23(i,4)*ones(1,length(procInd1_t23_Ia_in)),markAll_t23(procInd1_t23_I_in,4)',smker).*normpdf(markAll_t23(i,5)*ones(1,length(procInd1_t23_Ia_in)),markAll_t23(procInd1_t23_I_in,5)',smker);
                l1=Xnum_t23_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t23_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(17)==1 %tet27
                spike_r(17,t)=1;
                i=tet_sum(jj,17);
                l0=normpdf(markAll_t27(i,2)*ones(1,length(procInd1_t27_Ia_in)),markAll_t27(procInd1_t27_I_in,2)',smker).*normpdf(markAll_t27(i,3)*ones(1,length(procInd1_t27_Ia_in)),markAll_t27(procInd1_t27_I_in,3)',smker).*normpdf(markAll_t27(i,4)*ones(1,length(procInd1_t27_Ia_in)),markAll_t27(procInd1_t27_I_in,4)',smker).*normpdf(markAll_t27(i,5)*ones(1,length(procInd1_t27_Ia_in)),markAll_t27(procInd1_t27_I_in,5)',smker);
                l1=Xnum_t27_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t27_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            elseif tetVec(18)==1 %tet29
                spike_r(18,t)=1;
                i=tet_sum(jj,18);
                l0=normpdf(markAll_t29(i,2)*ones(1,length(procInd1_t29_Ia_in)),markAll_t29(procInd1_t29_I_in,2)',smker).*normpdf(markAll_t29(i,3)*ones(1,length(procInd1_t29_Ia_in)),markAll_t29(procInd1_t29_I_in,3)',smker).*normpdf(markAll_t29(i,4)*ones(1,length(procInd1_t29_Ia_in)),markAll_t29(procInd1_t29_I_in,4)',smker).*normpdf(markAll_t29(i,5)*ones(1,length(procInd1_t29_Ia_in)),markAll_t29(procInd1_t29_I_in,5)',smker);
                l1=Xnum_t29_I_in*l0'./occ_I_in(:,1)./dt;
                l2=l1.*dt.*exp(-Lint_t29_I_in.*dt);
                l2=l2./sum(l2);
                l_in(:,j)=l2;
            end
        end
        L_in=prod(l_in,2);L_in=L_in./sum(L_in);

        L_I0=L_out;L_I1=L_out;L_I2=L_in;L_I3=L_in;
    end
    
    totnorm=sum(onestep_I0.*L_I0)+sum(onestep_I1.*L_I1)+sum(onestep_I2.*L_I2)+sum(onestep_I3.*L_I3);
    postx_I0=onestep_I0.*L_I0./totnorm;
    postx_I1=onestep_I1.*L_I1./totnorm;
    postx_I2=onestep_I2.*L_I2./totnorm;
    postx_I3=onestep_I3.*L_I3./totnorm;

    pI0_vec(t)=sum(postx_I0);
    pI1_vec(t)=sum(postx_I1);
    pI2_vec(t)=sum(postx_I2);
    pI3_vec(t)=sum(postx_I3);
end

sumStat{pic}=[pI0_vec pI1_vec pI2_vec pI3_vec];
end
