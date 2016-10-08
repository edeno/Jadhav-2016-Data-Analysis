function [tetrode_index, neuron_index] = get_tetrodes_with_spikes(spikes, day, epoch, tetrode_number)
tetrode_index=[];neuron_index=[];
%this is from sorted spikes, we only use this to select replay events with
%non-zero sorted spikes in it to decode
for neuron_ind=1:length(tetrode_number)
    numNeurons=length(spikes{day}{epoch}{tetrode_number(neuron_ind)});
    b0=zeros(1,numNeurons);a0=zeros(1,numNeurons);
    if isempty(spikes{day}{epoch}{tetrode_number(neuron_ind)})==0
        for neuron_ind=1:numNeurons
            if ~isempty(spikes{day}{epoch}{tetrode_number(neuron_ind)}{neuron_ind}) && ...
                    ~isempty(spikes{day}{epoch}{tetrode_number(neuron_ind)}{neuron_ind}.data)
                b0(neuron_ind)=neuron_ind;
                a0(neuron_ind)=tetrode_number(neuron_ind);
            end
        end
    end
    neuron_index=[neuron_index b0(b0 > 0)];tetrode_index=[tetrode_index a0(a0 > 0)];
end
end