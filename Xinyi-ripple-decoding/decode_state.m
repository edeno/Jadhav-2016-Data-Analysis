function [summary_statistic] = decode_state(pos, ...
    rippleI, ...
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
    markAll, ...
    procInd1_Ia_out, ...
    procInd1_I_out, ...
    smker, ...
    Xnum_I_out, ....
    occ_Indicator_outbound, ...
    Lint_I_out, ...
    procInd1_Ia_in, ...
    procInd1_I_in, ...
    Xnum_I_in, ...
    occ_Indicator_inbound, ...
    Lint_I_in ...
    )

velocity=pos.data(:,5);
%linVel=linpos{ex}{ep}.statematrix.linearVelocity;
for pic=1:length(rippleI)
    rIndV=pic; %5, 12
    rloc_Ind=find(position_time_stamps*1000>position_time_stamps_binned(ripple_index(rippleI(rIndV),1))&position_time_stamps*1000<position_time_stamps_binned(ripple_index(rippleI(rIndV),2)));

    rloc(pic)=vecLF(rloc_Ind(1),2);
    vel(pic)=velocity(rloc_Ind(1),1);
end

velocity_threshold_index=find(vel<4);length(velocity_threshold_index)
%only decode replay when the running speed < 4cm/sec
ripplesconsN=traj_Ind(rippleI(velocity_threshold_index));

%% decoder
for pic=1:length(velocity_threshold_index)
    rIndV=velocity_threshold_index(pic); %5, 12

    spike_tim=ripple_index(rippleI(rIndV),1):ripple_index(rippleI(rIndV),2); %from 1 to 90000~
    numSteps=length(spike_tim);
    xi=round(time/10);

    %%
    dt=1/33.4;spike_r=zeros(18,numSteps);
    stateV_length=length(stateV);
    numSteps=size(spike_r,2);
    %P(x0|I);
    Px_I_out=exp(-stateV.^2./(2*(2*stateV_delta)^2));
    Px_I_out=Px_I_out./sum(Px_I_out);
    Px_I_in=max(Px_I_out)*ones(1,stateV_length)-Px_I_out;
    Px_I_in=Px_I_in./sum(Px_I_in);
    Px_I0=Px_I_out;
    Px_I1=Px_I_in;
    Px_I2=Px_I_in;
    Px_I3=Px_I_out;
    %P(x0)=P(x0|I)P(I);
    postx_I0=0.25*Px_I_out';
    postx_I1=0.25*Px_I_in';
    postx_I2=0.25*Px_I_in';
    postx_I3=0.25*Px_I_out';
    pI0_vec=zeros(numSteps,1);
    pI1_vec=zeros(numSteps,1);
    pI2_vec=zeros(numSteps,1);
    pI3_vec=zeros(numSteps,1);
    postxM_r_I0=zeros(stateV_length,numSteps);
    postxM_r_I1=zeros(stateV_length,numSteps);
    postxM_r_I2=zeros(stateV_length,numSteps);
    postxM_r_I3=zeros(stateV_length,numSteps);
    %state transition
    stateM_Indicator_outbound=stateM_I1_normalized_gaussian;
    stateM_Indicator_inbound=stateM_Indicator0_normalized_gaussian;
    stateM_I0=stateM_Indicator_outbound;
    stateM_I1=stateM_Indicator_inbound;
    stateM_I2=stateM_Indicator_inbound;
    stateM_I3=stateM_Indicator_outbound;
 
    for t=1:numSteps
        tt=spike_tim(t);
        aa=find(xi==position_time_stamps_binned(tt));

        onestep_I0=stateM_I0*postx_I0;
        onestep_I1=stateM_I1*postx_I1;
        onestep_I2=stateM_I2*postx_I2;
        onestep_I3=stateM_I3*postx_I3;

        L_I0=ones(stateV_length,1);L_I1=ones(stateV_length,1);L_I2=ones(stateV_length,1);L_I3=ones(stateV_length,1);

        if isempty(aa)==1 %if no spike occurs at time t
            L_I0=exp(-Lint_Indicator_outbound.*dt);L_I1=exp(-Lint_Indicator_outbound.*dt);
            L_I2=exp(-Lint_Indicator_inbound.*dt);L_I3=exp(-Lint_Indicator_inbound.*dt);

        elseif isempty(aa)==0 %if spikes

            l_out=zeros(stateV_length,length(aa));
            for j=1:length(aa)
                jj=aa(j);
                tetVec=tet_ind(jj,:);

                if tetVec(1)==1 %tet1
                    spike_r(1,t)=1;
                    i=tet_sum(jj,1);
                    l0=normpdf(markAll{1}(i,2)*ones(1,length(procInd1_Ia_out{1})),markAll{1}(procInd1_I_out{1},2)',smker).*normpdf(markAll{1}(i,3)*ones(1,length(procInd1_Ia_out{1})),markAll{1}(procInd1_I_out{1},3)',smker).*normpdf(markAll{1}(i,4)*ones(1,length(procInd1_Ia_out{1})),markAll{1}(procInd1_I_out{1},4)',smker).*normpdf(markAll{1}(i,5)*ones(1,length(procInd1_Ia_out{1})),markAll{1}(procInd1_I_out{1},5)',smker);
                    l1=Xnum_I_out{1}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{1}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(2)==1 %tet2
                    spike_r(2,t)=1;
                    i=tet_sum(jj,2);
                    l0=normpdf(markAll{2}(i,2)*ones(1,length(procInd1_Ia_out{2})),markAll{2}(procInd1_I_out{2},2)',smker).*normpdf(markAll{2}(i,3)*ones(1,length(procInd1_Ia_out{2})),markAll{2}(procInd1_I_out{2},3)',smker).*normpdf(markAll{2}(i,4)*ones(1,length(procInd1_Ia_out{2})),markAll{2}(procInd1_I_out{2},4)',smker).*normpdf(markAll{2}(i,5)*ones(1,length(procInd1_Ia_out{2})),markAll{2}(procInd1_I_out{2},5)',smker);
                    l1=Xnum_I_out{2}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{2}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(3)==1 %tet4
                    spike_r(3,t)=1;
                    i=tet_sum(jj,3);
                    l0=normpdf(markAll{3}(i,2)*ones(1,length(procInd1_Ia_out{3})),markAll{3}(procInd1_I_out{3},2)',smker).*normpdf(markAll{3}(i,3)*ones(1,length(procInd1_Ia_out{3})),markAll{3}(procInd1_I_out{3},3)',smker).*normpdf(markAll{3}(i,4)*ones(1,length(procInd1_Ia_out{3})),markAll{3}(procInd1_I_out{3},4)',smker).*normpdf(markAll{3}(i,5)*ones(1,length(procInd1_Ia_out{3})),markAll{3}(procInd1_I_out{3},5)',smker);
                    l1=Xnum_I_out{3}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{3}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(4)==1 %tet5
                    spike_r(4,t)=1;
                    i=tet_sum(jj,4);
                    l0=normpdf(markAll{4}(i,2)*ones(1,length(procInd1_Ia_out{4})),markAll{4}(procInd1_I_out{4},2)',smker).*normpdf(markAll{4}(i,3)*ones(1,length(procInd1_Ia_out{4})),markAll{4}(procInd1_I_out{4},3)',smker).*normpdf(markAll{4}(i,4)*ones(1,length(procInd1_Ia_out{4})),markAll{4}(procInd1_I_out{4},4)',smker).*normpdf(markAll{4}(i,5)*ones(1,length(procInd1_Ia_out{4})),markAll{4}(procInd1_I_out{4},5)',smker);
                    l1=Xnum_I_out{4}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{4}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(5)==1 %tet7
                    spike_r(5,t)=1;
                    i=tet_sum(jj,5);
                    l0=normpdf(markAll{5}(i,2)*ones(1,length(procInd1_Ia_out{5})),markAll{5}(procInd1_I_out{5},2)',smker).*normpdf(markAll{5}(i,3)*ones(1,length(procInd1_Ia_out{5})),markAll{5}(procInd1_I_out{5},3)',smker).*normpdf(markAll{5}(i,4)*ones(1,length(procInd1_Ia_out{5})),markAll{5}(procInd1_I_out{5},4)',smker).*normpdf(markAll{5}(i,5)*ones(1,length(procInd1_Ia_out{5})),markAll{5}(procInd1_I_out{5},5)',smker);
                    l1=Xnum_I_out{5}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{5}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(6)==1 %tet10
                    spike_r(6,t)=1;
                    i=tet_sum(jj,6);
                    l0=normpdf(markAll{6}(i,2)*ones(1,length(procInd1_Ia_out{6})),markAll{6}(procInd1_I_out{6},2)',smker).*normpdf(markAll{6}(i,3)*ones(1,length(procInd1_Ia_out{6})),markAll{6}(procInd1_I_out{6},3)',smker).*normpdf(markAll{6}(i,4)*ones(1,length(procInd1_Ia_out{6})),markAll{6}(procInd1_I_out{6},4)',smker).*normpdf(markAll{6}(i,5)*ones(1,length(procInd1_Ia_out{6})),markAll{6}(procInd1_I_out{6},5)',smker);
                    l1=Xnum_I_out{6}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{6}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(7)==1 %tet11
                    spike_r(7,t)=1;
                    i=tet_sum(jj,7);
                    l0=normpdf(markAll{7}(i,2)*ones(1,length(procInd1_Ia_out{7})),markAll{7}(procInd1_I_out{7},2)',smker).*normpdf(markAll{7}(i,3)*ones(1,length(procInd1_Ia_out{7})),markAll{7}(procInd1_I_out{7},3)',smker).*normpdf(markAll{7}(i,4)*ones(1,length(procInd1_Ia_out{7})),markAll{7}(procInd1_I_out{7},4)',smker).*normpdf(markAll{7}(i,5)*ones(1,length(procInd1_Ia_out{7})),markAll{7}(procInd1_I_out{7},5)',smker);
                    l1=Xnum_I_out{7}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{7}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(8)==1 %tet12
                    spike_r(8,t)=1;
                    i=tet_sum(jj,8);
                    l0=normpdf(markAll{8}(i,2)*ones(1,length(procInd1_Ia_out{8})),markAll{8}(procInd1_I_out{8},2)',smker).*normpdf(markAll{8}(i,3)*ones(1,length(procInd1_Ia_out{8})),markAll{8}(procInd1_I_out{8},3)',smker).*normpdf(markAll{8}(i,4)*ones(1,length(procInd1_Ia_out{8})),markAll{8}(procInd1_I_out{8},4)',smker).*normpdf(markAll{8}(i,5)*ones(1,length(procInd1_Ia_out{8})),markAll{8}(procInd1_I_out{8},5)',smker);
                    l1=Xnum_I_out{8}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{8}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(9)==1 %tet13
                    spike_r(9,t)=1;
                    i=tet_sum(jj,9);
                    l0=normpdf(markAll{9}(i,2)*ones(1,length(procInd1_Ia_out{9})),markAll{9}(procInd1_I_out{9},2)',smker).*normpdf(markAll{9}(i,3)*ones(1,length(procInd1_Ia_out{9})),markAll{9}(procInd1_I_out{9},3)',smker).*normpdf(markAll{9}(i,4)*ones(1,length(procInd1_Ia_out{9})),markAll{9}(procInd1_I_out{9},4)',smker).*normpdf(markAll{9}(i,5)*ones(1,length(procInd1_Ia_out{9})),markAll{9}(procInd1_I_out{9},5)',smker);
                    l1=Xnum_I_out{9}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{9}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(10)==1 %tet14
                    spike_r(10,t)=1;
                    i=tet_sum(jj,10);
                    l0=normpdf(markAll{10}(i,2)*ones(1,length(procInd1_Ia_out{10})),markAll{10}(procInd1_I_out{10},2)',smker).*normpdf(markAll{10}(i,3)*ones(1,length(procInd1_Ia_out{10})),markAll{10}(procInd1_I_out{10},3)',smker).*normpdf(markAll{10}(i,4)*ones(1,length(procInd1_Ia_out{10})),markAll{10}(procInd1_I_out{10},4)',smker).*normpdf(markAll{10}(i,5)*ones(1,length(procInd1_Ia_out{10})),markAll{10}(procInd1_I_out{10},5)',smker);
                    l1=Xnum_I_out{10}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{10}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(11)==1 %tet17
                    spike_r(11,t)=1;
                    i=tet_sum(jj,11);
                    l0=normpdf(markAll{11}(i,2)*ones(1,length(procInd1_Ia_out{11})),markAll{11}(procInd1_I_out{11},2)',smker).*normpdf(markAll{11}(i,3)*ones(1,length(procInd1_Ia_out{11})),markAll{11}(procInd1_I_out{11},3)',smker).*normpdf(markAll{11}(i,4)*ones(1,length(procInd1_Ia_out{11})),markAll{11}(procInd1_I_out{11},4)',smker).*normpdf(markAll{11}(i,5)*ones(1,length(procInd1_Ia_out{11})),markAll{11}(procInd1_I_out{11},5)',smker);
                    l1=Xnum_I_out{11}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{11}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(12)==1 %tet18
                    spike_r(12,t)=1;
                    i=tet_sum(jj,12);
                    l0=normpdf(markAll{12}(i,2)*ones(1,length(procInd1_Ia_out{12})),markAll{12}(procInd1_I_out{12},2)',smker).*normpdf(markAll{12}(i,3)*ones(1,length(procInd1_Ia_out{12})),markAll{12}(procInd1_I_out{12},3)',smker).*normpdf(markAll{12}(i,4)*ones(1,length(procInd1_Ia_out{12})),markAll{12}(procInd1_I_out{12},4)',smker).*normpdf(markAll{12}(i,5)*ones(1,length(procInd1_Ia_out{12})),markAll{12}(procInd1_I_out{12},5)',smker);
                    l1=Xnum_I_out{12}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{12}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(13)==1 %tet19
                    spike_r(13,t)=1;
                    i=tet_sum(jj,13);
                    l0=normpdf(markAll{13}(i,2)*ones(1,length(procInd1_Ia_out{13})),markAll{13}(procInd1_I_out{13},2)',smker).*normpdf(markAll{13}(i,3)*ones(1,length(procInd1_Ia_out{13})),markAll{13}(procInd1_I_out{13},3)',smker).*normpdf(markAll{13}(i,4)*ones(1,length(procInd1_Ia_out{13})),markAll{13}(procInd1_I_out{13},4)',smker).*normpdf(markAll{13}(i,5)*ones(1,length(procInd1_Ia_out{13})),markAll{13}(procInd1_I_out{13},5)',smker);
                    l1=Xnum_I_out{13}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{13}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(14)==1 %tet20
                    spike_r(14,t)=1;
                    i=tet_sum(jj,14);
                    l0=normpdf(markAll{14}(i,2)*ones(1,length(procInd1_Ia_out{14})),markAll{14}(procInd1_I_out{14},2)',smker).*normpdf(markAll{14}(i,3)*ones(1,length(procInd1_Ia_out{14})),markAll{14}(procInd1_I_out{14},3)',smker).*normpdf(markAll{14}(i,4)*ones(1,length(procInd1_Ia_out{14})),markAll{14}(procInd1_I_out{14},4)',smker).*normpdf(markAll{14}(i,5)*ones(1,length(procInd1_Ia_out{14})),markAll{14}(procInd1_I_out{14},5)',smker);
                    l1=Xnum_I_out{14}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{14}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(15)==1 %tet22
                    spike_r(15,t)=1;
                    i=tet_sum(jj,15);
                    l0=normpdf(markAll{15}(i,2)*ones(1,length(procInd1_Ia_out{15})),markAll{15}(procInd1_I_out{15},2)',smker).*normpdf(markAll{15}(i,3)*ones(1,length(procInd1_Ia_out{15})),markAll{15}(procInd1_I_out{15},3)',smker).*normpdf(markAll{15}(i,4)*ones(1,length(procInd1_Ia_out{15})),markAll{15}(procInd1_I_out{15},4)',smker).*normpdf(markAll{15}(i,5)*ones(1,length(procInd1_Ia_out{15})),markAll{15}(procInd1_I_out{15},5)',smker);
                    l1=Xnum_I_out{15}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{15}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(16)==1 %tet23
                    spike_r(16,t)=1;
                    i=tet_sum(jj,16);
                    l0=normpdf(markAll{16}(i,2)*ones(1,length(procInd1_Ia_out{16})),markAll{16}(procInd1_I_out{16},2)',smker).*normpdf(markAll{16}(i,3)*ones(1,length(procInd1_Ia_out{16})),markAll{16}(procInd1_I_out{16},3)',smker).*normpdf(markAll{16}(i,4)*ones(1,length(procInd1_Ia_out{16})),markAll{16}(procInd1_I_out{16},4)',smker).*normpdf(markAll{16}(i,5)*ones(1,length(procInd1_Ia_out{16})),markAll{16}(procInd1_I_out{16},5)',smker);
                    l1=Xnum_I_out{16}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{16}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(17)==1 %tet27
                    spike_r(17,t)=1;
                    i=tet_sum(jj,17);
                    l0=normpdf(markAll{17}(i,2)*ones(1,length(procInd1_Ia_out{17})),markAll{17}(procInd1_I_out{17},2)',smker).*normpdf(markAll{17}(i,3)*ones(1,length(procInd1_Ia_out{17})),markAll{17}(procInd1_I_out{17},3)',smker).*normpdf(markAll{17}(i,4)*ones(1,length(procInd1_Ia_out{17})),markAll{17}(procInd1_I_out{17},4)',smker).*normpdf(markAll{17}(i,5)*ones(1,length(procInd1_Ia_out{17})),markAll{17}(procInd1_I_out{17},5)',smker);
                    l1=Xnum_I_out{17}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{17}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                elseif tetVec(18)==1 %tet29
                    spike_r(18,t)=1;
                    i=tet_sum(jj,18);
                    l0=normpdf(markAll{18}(i,2)*ones(1,length(procInd1_Ia_out{18})),markAll{18}(procInd1_I_out{18},2)',smker).*normpdf(markAll{18}(i,3)*ones(1,length(procInd1_Ia_out{18})),markAll{18}(procInd1_I_out{18},3)',smker).*normpdf(markAll{18}(i,4)*ones(1,length(procInd1_Ia_out{18})),markAll{18}(procInd1_I_out{18},4)',smker).*normpdf(markAll{18}(i,5)*ones(1,length(procInd1_Ia_out{18})),markAll{18}(procInd1_I_out{18},5)',smker);
                    l1=Xnum_I_out{18}*l0'./occ_Indicator_outbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_out{18}.*dt);
                    l2=l2./sum(l2);
                    l_out(:,j)=l2;
                end
            end
            L_out=prod(l_out,2);L_out=L_out./sum(L_out);

            l_in=zeros(stateV_length,length(aa));
            for j=1:length(aa)
                jj=aa(j);
                tetVec=tet_ind(jj,:);

                if tetVec(1)==1 %tet1
                    spike_r(1,t)=1;
                    i=tet_sum(jj,1);
                    l0=normpdf(markAll{1}(i,2)*ones(1,length(procInd1_Ia_in{1})),markAll{1}(procInd1_I_in{1},2)',smker).*normpdf(markAll{1}(i,3)*ones(1,length(procInd1_Ia_in{1})),markAll{1}(procInd1_I_in{1},3)',smker).*normpdf(markAll{1}(i,4)*ones(1,length(procInd1_Ia_in{1})),markAll{1}(procInd1_I_in{1},4)',smker).*normpdf(markAll{1}(i,5)*ones(1,length(procInd1_Ia_in{1})),markAll{1}(procInd1_I_in{1},5)',smker);
                    l1=Xnum_I_in{1}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{1}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(2)==1 %tet2
                    spike_r(2,t)=1;
                    i=tet_sum(jj,2);
                    l0=normpdf(markAll{2}(i,2)*ones(1,length(procInd1_Ia_in{2})),markAll{2}(procInd1_I_in{2},2)',smker).*normpdf(markAll{2}(i,3)*ones(1,length(procInd1_Ia_in{2})),markAll{2}(procInd1_I_in{2},3)',smker).*normpdf(markAll{2}(i,4)*ones(1,length(procInd1_Ia_in{2})),markAll{2}(procInd1_I_in{2},4)',smker).*normpdf(markAll{2}(i,5)*ones(1,length(procInd1_Ia_in{2})),markAll{2}(procInd1_I_in{2},5)',smker);
                    l1=Xnum_I_in{2}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{2}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(3)==1 %tet4
                    spike_r(3,t)=1;
                    i=tet_sum(jj,3);
                    l0=normpdf(markAll{3}(i,2)*ones(1,length(procInd1_Ia_in{3})),markAll{3}(procInd1_I_in{3},2)',smker).*normpdf(markAll{3}(i,3)*ones(1,length(procInd1_Ia_in{3})),markAll{3}(procInd1_I_in{3},3)',smker).*normpdf(markAll{3}(i,4)*ones(1,length(procInd1_Ia_in{3})),markAll{3}(procInd1_I_in{3},4)',smker).*normpdf(markAll{3}(i,5)*ones(1,length(procInd1_Ia_in{3})),markAll{3}(procInd1_I_in{3},5)',smker);
                    l1=Xnum_I_in{3}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{3}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(4)==1 %tet5
                    spike_r(4,t)=1;
                    i=tet_sum(jj,4);
                    l0=normpdf(markAll{4}(i,2)*ones(1,length(procInd1_Ia_in{4})),markAll{4}(procInd1_I_in{4},2)',smker).*normpdf(markAll{4}(i,3)*ones(1,length(procInd1_Ia_in{4})),markAll{4}(procInd1_I_in{4},3)',smker).*normpdf(markAll{4}(i,4)*ones(1,length(procInd1_Ia_in{4})),markAll{4}(procInd1_I_in{4},4)',smker).*normpdf(markAll{4}(i,5)*ones(1,length(procInd1_Ia_in{4})),markAll{4}(procInd1_I_in{4},5)',smker);
                    l1=Xnum_I_in{4}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{4}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(5)==1 %tet7
                    spike_r(5,t)=1;
                    i=tet_sum(jj,5);
                    l0=normpdf(markAll{5}(i,2)*ones(1,length(procInd1_Ia_in{5})),markAll{5}(procInd1_I_in{5},2)',smker).*normpdf(markAll{5}(i,3)*ones(1,length(procInd1_Ia_in{5})),markAll{5}(procInd1_I_in{5},3)',smker).*normpdf(markAll{5}(i,4)*ones(1,length(procInd1_Ia_in{5})),markAll{5}(procInd1_I_in{5},4)',smker).*normpdf(markAll{5}(i,5)*ones(1,length(procInd1_Ia_in{5})),markAll{5}(procInd1_I_in{5},5)',smker);
                    l1=Xnum_I_in{5}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{5}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(6)==1 %tet10
                    spike_r(6,t)=1;
                    i=tet_sum(jj,6);
                    l0=normpdf(markAll{6}(i,2)*ones(1,length(procInd1_Ia_in{6})),markAll{6}(procInd1_I_in{6},2)',smker).*normpdf(markAll{6}(i,3)*ones(1,length(procInd1_Ia_in{6})),markAll{6}(procInd1_I_in{6},3)',smker).*normpdf(markAll{6}(i,4)*ones(1,length(procInd1_Ia_in{6})),markAll{6}(procInd1_I_in{6},4)',smker).*normpdf(markAll{6}(i,5)*ones(1,length(procInd1_Ia_in{6})),markAll{6}(procInd1_I_in{6},5)',smker);
                    l1=Xnum_I_in{6}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{6}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(7)==1 %tet11
                    spike_r(7,t)=1;
                    i=tet_sum(jj,7);
                    l0=normpdf(markAll{7}(i,2)*ones(1,length(procInd1_Ia_in{7})),markAll{7}(procInd1_I_in{7},2)',smker).*normpdf(markAll{7}(i,3)*ones(1,length(procInd1_Ia_in{7})),markAll{7}(procInd1_I_in{7},3)',smker).*normpdf(markAll{7}(i,4)*ones(1,length(procInd1_Ia_in{7})),markAll{7}(procInd1_I_in{7},4)',smker).*normpdf(markAll{7}(i,5)*ones(1,length(procInd1_Ia_in{7})),markAll{7}(procInd1_I_in{7},5)',smker);
                    l1=Xnum_I_in{7}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{7}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(8)==1 %tet12
                    spike_r(8,t)=1;
                    i=tet_sum(jj,8);
                    l0=normpdf(markAll{8}(i,2)*ones(1,length(procInd1_Ia_in{8})),markAll{8}(procInd1_I_in{8},2)',smker).*normpdf(markAll{8}(i,3)*ones(1,length(procInd1_Ia_in{8})),markAll{8}(procInd1_I_in{8},3)',smker).*normpdf(markAll{8}(i,4)*ones(1,length(procInd1_Ia_in{8})),markAll{8}(procInd1_I_in{8},4)',smker).*normpdf(markAll{8}(i,5)*ones(1,length(procInd1_Ia_in{8})),markAll{8}(procInd1_I_in{8},5)',smker);
                    l1=Xnum_I_in{8}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{8}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(9)==1 %tet13
                    spike_r(9,t)=1;
                    i=tet_sum(jj,9);
                    l0=normpdf(markAll{9}(i,2)*ones(1,length(procInd1_Ia_in{9})),markAll{9}(procInd1_I_in{9},2)',smker).*normpdf(markAll{9}(i,3)*ones(1,length(procInd1_Ia_in{9})),markAll{9}(procInd1_I_in{9},3)',smker).*normpdf(markAll{9}(i,4)*ones(1,length(procInd1_Ia_in{9})),markAll{9}(procInd1_I_in{9},4)',smker).*normpdf(markAll{9}(i,5)*ones(1,length(procInd1_Ia_in{9})),markAll{9}(procInd1_I_in{9},5)',smker);
                    l1=Xnum_I_in{9}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{9}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(10)==1 %tet14
                    spike_r(10,t)=1;
                    i=tet_sum(jj,10);
                    l0=normpdf(markAll{10}(i,2)*ones(1,length(procInd1_Ia_in{10})),markAll{10}(procInd1_I_in{10},2)',smker).*normpdf(markAll{10}(i,3)*ones(1,length(procInd1_Ia_in{10})),markAll{10}(procInd1_I_in{10},3)',smker).*normpdf(markAll{10}(i,4)*ones(1,length(procInd1_Ia_in{10})),markAll{10}(procInd1_I_in{10},4)',smker).*normpdf(markAll{10}(i,5)*ones(1,length(procInd1_Ia_in{10})),markAll{10}(procInd1_I_in{10},5)',smker);
                    l1=Xnum_I_in{10}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{10}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(11)==1 %tet17
                    spike_r(11,t)=1;
                    i=tet_sum(jj,11);
                    l0=normpdf(markAll{11}(i,2)*ones(1,length(procInd1_Ia_in{11})),markAll{11}(procInd1_I_in{11},2)',smker).*normpdf(markAll{11}(i,3)*ones(1,length(procInd1_Ia_in{11})),markAll{11}(procInd1_I_in{11},3)',smker).*normpdf(markAll{11}(i,4)*ones(1,length(procInd1_Ia_in{11})),markAll{11}(procInd1_I_in{11},4)',smker).*normpdf(markAll{11}(i,5)*ones(1,length(procInd1_Ia_in{11})),markAll{11}(procInd1_I_in{11},5)',smker);
                    l1=Xnum_I_in{11}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{11}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(12)==1 %tet18
                    spike_r(12,t)=1;
                    i=tet_sum(jj,12);
                    l0=normpdf(markAll{12}(i,2)*ones(1,length(procInd1_Ia_in{12})),markAll{12}(procInd1_I_in{12},2)',smker).*normpdf(markAll{12}(i,3)*ones(1,length(procInd1_Ia_in{12})),markAll{12}(procInd1_I_in{12},3)',smker).*normpdf(markAll{12}(i,4)*ones(1,length(procInd1_Ia_in{12})),markAll{12}(procInd1_I_in{12},4)',smker).*normpdf(markAll{12}(i,5)*ones(1,length(procInd1_Ia_in{12})),markAll{12}(procInd1_I_in{12},5)',smker);
                    l1=Xnum_I_in{12}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{12}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(13)==1 %tet19
                    spike_r(13,t)=1;
                    i=tet_sum(jj,13);
                    l0=normpdf(markAll{13}(i,2)*ones(1,length(procInd1_Ia_in{13})),markAll{13}(procInd1_I_in{13},2)',smker).*normpdf(markAll{13}(i,3)*ones(1,length(procInd1_Ia_in{13})),markAll{13}(procInd1_I_in{13},3)',smker).*normpdf(markAll{13}(i,4)*ones(1,length(procInd1_Ia_in{13})),markAll{13}(procInd1_I_in{13},4)',smker).*normpdf(markAll{13}(i,5)*ones(1,length(procInd1_Ia_in{13})),markAll{13}(procInd1_I_in{13},5)',smker);
                    l1=Xnum_I_in{13}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{13}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(14)==1 %tet20
                    spike_r(14,t)=1;
                    i=tet_sum(jj,14);
                    l0=normpdf(markAll{14}(i,2)*ones(1,length(procInd1_Ia_in{14})),markAll{14}(procInd1_I_in{14},2)',smker).*normpdf(markAll{14}(i,3)*ones(1,length(procInd1_Ia_in{14})),markAll{14}(procInd1_I_in{14},3)',smker).*normpdf(markAll{14}(i,4)*ones(1,length(procInd1_Ia_in{14})),markAll{14}(procInd1_I_in{14},4)',smker).*normpdf(markAll{14}(i,5)*ones(1,length(procInd1_Ia_in{14})),markAll{14}(procInd1_I_in{14},5)',smker);
                    l1=Xnum_I_in{14}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{14}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(15)==1 %tet22
                    spike_r(15,t)=1;
                    i=tet_sum(jj,15);
                    l0=normpdf(markAll{15}(i,2)*ones(1,length(procInd1_Ia_in{15})),markAll{15}(procInd1_I_in{15},2)',smker).*normpdf(markAll{15}(i,3)*ones(1,length(procInd1_Ia_in{15})),markAll{15}(procInd1_I_in{15},3)',smker).*normpdf(markAll{15}(i,4)*ones(1,length(procInd1_Ia_in{15})),markAll{15}(procInd1_I_in{15},4)',smker).*normpdf(markAll{15}(i,5)*ones(1,length(procInd1_Ia_in{15})),markAll{15}(procInd1_I_in{15},5)',smker);
                    l1=Xnum_I_in{15}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{15}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(16)==1 %tet23
                    spike_r(16,t)=1;
                    i=tet_sum(jj,16);
                    l0=normpdf(markAll{16}(i,2)*ones(1,length(procInd1_Ia_in{16})),markAll{16}(procInd1_I_in{16},2)',smker).*normpdf(markAll{16}(i,3)*ones(1,length(procInd1_Ia_in{16})),markAll{16}(procInd1_I_in{16},3)',smker).*normpdf(markAll{16}(i,4)*ones(1,length(procInd1_Ia_in{16})),markAll{16}(procInd1_I_in{16},4)',smker).*normpdf(markAll{16}(i,5)*ones(1,length(procInd1_Ia_in{16})),markAll{16}(procInd1_I_in{16},5)',smker);
                    l1=Xnum_I_in{16}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{16}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(17)==1 %tet27
                    spike_r(17,t)=1;
                    i=tet_sum(jj,17);
                    l0=normpdf(markAll{17}(i,2)*ones(1,length(procInd1_Ia_in{17})),markAll{17}(procInd1_I_in{17},2)',smker).*normpdf(markAll{17}(i,3)*ones(1,length(procInd1_Ia_in{17})),markAll{17}(procInd1_I_in{17},3)',smker).*normpdf(markAll{17}(i,4)*ones(1,length(procInd1_Ia_in{17})),markAll{17}(procInd1_I_in{17},4)',smker).*normpdf(markAll{17}(i,5)*ones(1,length(procInd1_Ia_in{17})),markAll{17}(procInd1_I_in{17},5)',smker);
                    l1=Xnum_I_in{17}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{17}.*dt);
                    l2=l2./sum(l2);
                    l_in(:,j)=l2;
                elseif tetVec(18)==1 %tet29
                    spike_r(18,t)=1;
                    i=tet_sum(jj,18);
                    l0=normpdf(markAll{18}(i,2)*ones(1,length(procInd1_Ia_in{18})),markAll{18}(procInd1_I_in{18},2)',smker).*normpdf(markAll{18}(i,3)*ones(1,length(procInd1_Ia_in{18})),markAll{18}(procInd1_I_in{18},3)',smker).*normpdf(markAll{18}(i,4)*ones(1,length(procInd1_Ia_in{18})),markAll{18}(procInd1_I_in{18},4)',smker).*normpdf(markAll{18}(i,5)*ones(1,length(procInd1_Ia_in{18})),markAll{18}(procInd1_I_in{18},5)',smker);
                    l1=Xnum_I_in{18}*l0'./occ_Indicator_inbound(:,1)./dt;
                    l2=l1.*dt.*exp(-Lint_I_in{18}.*dt);
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

    summary_statistic{pic}=[pI0_vec pI1_vec pI2_vec pI3_vec];
end
end