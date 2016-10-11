function [state_index] = get_state_index(trajencode)
state_index{1} = find(trajencode.trajstate == 1 | trajencode.trajstate == 3); % outbound
state_index{2} = find(trajencode.trajstate == 2 | trajencode.trajstate == 4); % inbound
end